import torch
import PIL.Image
from typing import Optional, Tuple, List, Mapping, Dict, Union
from dataclasses import dataclass
from .tree import Tree, TreeOp
from .owl_predictor import OwlPredictor, OwlEncodeTextOutput, OwlEncodeImageOutput
from .clip_predictor2 import ClipPredictor, ClipEncodeTextOutput, ClipEncodeImageOutput
from .image_preprocessor import ImagePreprocessor

@dataclass
class TreeDetection:
    id: int
    parent_id: int
    box: Tuple[float, float, float, float]
    labels: List[int]
    scores: List[int]

@dataclass
class OwlDecodeOutput:
    labels: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor
    input_indices: torch.Tensor

@dataclass
class TreeOutput:
    detections: List[TreeDetection]

class TreePredictor(torch.nn.Module):

    def __init__(self,
                 owl_predictor: Optional[OwlPredictor] = None,
                 clip_predictor: Optional[ClipPredictor] = None,
                 image_preprocessor: Optional[ImagePreprocessor] = None,
                 device: str = "cuda"):
        super().__init__()
        self.owl_predictor = OwlPredictor() if owl_predictor is None else owl_predictor
        self.clip_predictor = ClipPredictor() if clip_predictor is None else clip_predictor
        self.image_preprocessor = ImagePreprocessor().to(device).eval() if image_preprocessor is None else image_preprocessor

    def encode_clip_text(self, tree: Tree) -> Dict[int, ClipEncodeTextOutput]:
        label_indices = tree.get_classify_label_indices()
        if len(label_indices) == 0:
            return {}
        labels = [tree.labels[index] for index in label_indices]
        text_encodings = self.clip_predictor.encode_text(labels)
        label_encodings = {}
        for i in range(len(labels)):
            label_encodings[label_indices[i]] = text_encodings.slice(i, i + 1)
        return label_encodings

    def encode_owl_text(self, tree: Tree) -> Dict[int, OwlEncodeTextOutput]:
        label_indices = tree.get_detect_label_indices()
        if len(label_indices) == 0:
            return {}
        labels = [tree.labels[index] for index in label_indices]
        text_encodings = self.owl_predictor.encode_text(labels)
        label_encodings = {}
        for i in range(len(labels)):
            label_encodings[label_indices[i]] = text_encodings.slice(i, i + 1)
        return label_encodings

    @torch.no_grad()
    def predict(self,
                image: Union[PIL.Image.Image, torch.Tensor],
                tree: Tree,
                threshold: float = 0.1,
                clip_text_encodings: Optional[Dict[int, ClipEncodeTextOutput]] = None,
                owl_text_encodings: Optional[Dict[int, OwlEncodeTextOutput]] = None):

        if clip_text_encodings is None:
            clip_text_encodings = self.encode_clip_text(tree)
        print("clip_text_encodings:", clip_text_encodings)

        if owl_text_encodings is None:
            owl_text_encodings = self.encode_owl_text(tree)
        print("owl_text_encodings:", owl_text_encodings)

        if isinstance(image, PIL.Image.Image):
            image_tensor = self.image_preprocessor.preprocess_pil_image(image)
            boxes = {
                0: torch.tensor([[0, 0, image.width, image.height]], dtype=image_tensor.dtype, device=image_tensor.device)
            }
        elif isinstance(image, torch.Tensor):
            image_tensor = self.image_preprocessor.preprocess_tensor_image(image)
            boxes = {
               0: torch.tensor([[0, 0, image.shape[2], image.shape[1]]], dtype=image_tensor.dtype, device=image_tensor.device)
            }
        else:
            raise ValueError("Input image must be either a PIL Image or a torch.Tensor")
        print("image_tensor shape:", image_tensor.shape)
        print("boxes:", boxes)

        scores = {
            0: torch.tensor([1.], dtype=torch.float, device=image_tensor.device)
        }
        instance_ids = {
            0: torch.tensor([0], dtype=torch.int64, device=image_tensor.device)
        }
        parent_instance_ids = {
            0: torch.tensor([-1], dtype=torch.int64, device=image_tensor.device)
        }
        print("Initial scores:", scores)
        print("Initial instance_ids:", instance_ids)
        print("Initial parent_instance_ids:", parent_instance_ids)

        owl_image_encodings: Dict[int, OwlEncodeImageOutput] = {}
        clip_image_encodings: Dict[int, ClipEncodeImageOutput] = {}

        global_instance_id = 1
        queue = [0]

        while queue:
            label_index = queue.pop(0)
            print("Processing label_index:", label_index)

            detect_nodes = tree.find_detect_nodes_with_input(label_index)
            classify_nodes = tree.find_classify_nodes_with_input(label_index)
            print("detect_nodes:", detect_nodes)
            print("classify_nodes:", classify_nodes)
            print(image_tensor.shape)
            print(boxes[label_index].shape)
            # Run OWL image encode if required
            if len(detect_nodes) > 0 and label_index not in owl_image_encodings:
                owl_image_encodings[label_index] = self.owl_predictor.encode_rois(image_tensor, boxes[label_index])
            # print("owl_image_encodings:", owl_image_encodings)

            # Run CLIP image encode if required
            if len(classify_nodes) > 0 and label_index not in clip_image_encodings:
                clip_image_encodings[label_index] = self.clip_predictor.encode_rois(image_tensor, boxes[label_index])
            # print("clip_image_encodings:", clip_image_encodings)

            # Decode detect nodes
            for node in detect_nodes:
                if node.input not in owl_image_encodings:
                    raise RuntimeError("Missing owl image encodings for node.")

                # gather encodings
                owl_text_encodings_for_node = OwlEncodeTextOutput(
                    text_embeds=torch.cat([
                        owl_text_encodings[i].text_embeds for i in node.outputs
                    ], dim=0)
                )
                # print("owl_text_encodings_for_node:", owl_text_encodings_for_node)

                owl_node_output = self.owl_predictor.decode(
                    owl_image_encodings[node.input],
                    owl_text_encodings_for_node,
                    threshold=threshold
                )
                # print("owl_node_output:", owl_node_output)

                num_detections = len(owl_node_output.labels)
                instance_ids_for_node = torch.arange(global_instance_id, global_instance_id + num_detections, dtype=torch.int64, device=owl_node_output.labels.device)
                parent_instance_ids_for_node = instance_ids[node.input][owl_node_output.input_indices]
                global_instance_id += num_detections
                print("num_detections:", num_detections)
                print("instance_ids_for_node:", instance_ids_for_node)
                print("parent_instance_ids_for_node:", parent_instance_ids_for_node)

                for i in range(len(node.outputs)):
                    mask = owl_node_output.labels == i
                    out_idx = node.outputs[i]
                    boxes[out_idx] = owl_node_output.boxes[mask]
                    scores[out_idx] = owl_node_output.scores[mask]
                    instance_ids[out_idx] = instance_ids_for_node[mask]
                    parent_instance_ids[out_idx] = parent_instance_ids_for_node[mask]
                    print(f"Updated boxes[{out_idx}]:", boxes[out_idx])
                    print(f"Updated scores[{out_idx}]:", scores[out_idx])
                    print(f"Updated instance_ids[{out_idx}]:", instance_ids[out_idx])
                    print(f"Updated parent_instance_ids[{out_idx}]:", parent_instance_ids[out_idx])

            for node in classify_nodes:
                if node.input not in clip_image_encodings:
                    raise RuntimeError("Missing clip image encodings for node.")

                clip_text_encodings_for_node = ClipEncodeTextOutput(
                    text_embeds=torch.cat([
                        clip_text_encodings[i].text_embeds for i in node.outputs
                    ], dim=0)
                )
                print("clip_text_encodings_for_node:", clip_text_encodings_for_node)

                clip_node_output = self.clip_predictor.decode(
                    clip_image_encodings[node.input],
                    clip_text_encodings_for_node
                )
                print("clip_node_output:", clip_node_output)

                parent_instance_ids_for_node = instance_ids[node.input]

                for i in range(len(node.outputs)):
                    mask = clip_node_output.labels == i
                    output_buffer = node.outputs[i]
                    scores[output_buffer] = clip_node_output.scores[mask].float()
                    boxes[output_buffer] = boxes[label_index][mask].float()
                    instance_ids[output_buffer] = instance_ids[node.input][mask]
                    parent_instance_ids[output_buffer] = parent_instance_ids[node.input][mask]
                    print(f"Updated boxes[{output_buffer}]:", boxes[output_buffer])
                    print(f"Updated scores[{output_buffer}]:", scores[output_buffer])
                    print(f"Updated instance_ids[{output_buffer}]:", instance_ids[output_buffer])
                    print(f"Updated parent_instance_ids[{output_buffer}]:", parent_instance_ids[output_buffer])

            for node in detect_nodes:
                for buf in node.outputs:
                    if buf in scores and len(scores[buf]) > 0:
                        queue.append(buf)
                        print(f"Added {buf} to queue")

            for node in classify_nodes:
                for buf in node.outputs:
                    if buf in scores and len(scores[buf]) > 0:
                        queue.append(buf)
                        print(f"Added {buf} to queue")

        # Fill outputs
        all_labels = []
        all_scores = []
        all_boxes = []
        all_parent_ids = []

        for i in boxes.keys():
            for box, score, instance_id, parent_instance_id in zip(boxes[i], scores[i], instance_ids[i], parent_instance_ids[i]):
                instance_id = int(instance_id)
                score = float(score)
                box = box.tolist()
                parent_instance_id = int(parent_instance_id)

                all_labels.append(i)
                all_scores.append(score)
                all_boxes.append(box)
                all_parent_ids.append(parent_instance_id)

        labels_tensor = torch.tensor(all_labels, dtype=torch.int64)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float16)
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float16)
        parent_ids_tensor = torch.tensor(all_parent_ids, dtype=torch.int64)

        print("Final labels_tensor:", labels_tensor)
        print("Final scores_tensor:", scores_tensor)
        print("Final boxes_tensor:", boxes_tensor)
        print("Final parent_ids_tensor:", parent_ids_tensor)

        return OwlDecodeOutput(
            labels=labels_tensor,
            scores=scores_tensor,
            boxes=boxes_tensor,
            input_indices=parent_ids_tensor
        )
