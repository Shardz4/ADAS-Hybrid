import queue
import threading
from typing import Optional
import numpy as np

class VLMEngine:
    """Asynchronous Vision-Language Model interface for real-time benchmark logging."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct"):
        self.model_name = model_name
        self._queue = queue.Queue(maxsize=1)
        self._latest_result: str = "VLM: Initializing..."
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray, prompt: str = "Identify immediate road hazards."):
        """Drop the frame into the processing queue without blocking the HUD loop."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            # Pass downscaled copy to keep memory footprint minimal
            small_frame = frame[::2, ::2].copy()
            self._queue.put_nowait((small_frame, prompt))
        except queue.Full:
            pass

    def get_latest_response(self) -> str:
        return self._latest_result

    def _worker(self):
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, AutoModelForVision2Seq

            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self._latest_result = "VLM: Ready"
        except Exception as e:
            self._latest_result = f"VLM Unavailable: {e}"
            return

        while not self._stop_event.is_set():
            try:
                frame, prompt = self._queue.get(timeout=0.2)
                image = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=45)
                    generated_texts = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                
                raw_out = generated_texts[0].split("Assistant:")[-1].strip()
                self._latest_result = f"VLM: {raw_out[:70]}"
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self._latest_result = f"VLM Error: {e}"

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)