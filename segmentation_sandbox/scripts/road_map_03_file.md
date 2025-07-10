# 🛠️ GroundedDINO High-Quality Annotation Integration Roadmap

## 📁 Target File
- `scripts/utils/grounded_sam_utils.py`

---

## 1. 🧠 Add Method: `generate_high_quality_annotations`

### 📍 Insert Location
Inside the `GroundedDinoAnnotations` class  
➡️ After or near `def process_missing_annotations(...)`

### 📌 Method Signature & Purpose
```python
def generate_high_quality_annotations(self,
                                      image_ids: List[str],
                                      prompt: str = "individual embryo",
                                      confidence_threshold: float = 0.5,
                                      iou_threshold: float = 0.5,
                                      overwrite: bool = False,
                                      save_to_self: bool = True) -> Dict:
```

### 🔧 Logic Overview

#### ✅ Step 1: Filter annotations by prompt and image_ids
- Iterate through `self.annotations["images"]`
- Retain only entries where `image_id in image_ids` and prompt matches

#### ✅ Step 2: Filter by confidence
- Keep detections with `confidence >= confidence_threshold`

#### ✅ Step 3: Filter by IoU
- Use existing logic (reference `03_initial_gdino_detections.py` lines **185–228**)  
- Apply Non-Max Suppression by IoU within each `image_id`

#### ✅ Step 4: Group by experiment
- Use `self._metadata` to map each `image_id` to its experiment (via `image_to_experiment_map = {...}`)

#### ✅ Step 5: Optionally save results to self
```python
if save_to_self:
    self.annotations["high_quality_annotations"][exp_id] = {
        ...
    }
    self._unsaved_changes = True
```

#### ✅ Step 6: Return results dictionary (always returned)

---

## 2. 💡 Add Helper: `get_or_generate_high_quality_annotations`

### 📍 Purpose
Return high quality annotations for a given config on-the-fly.
Avoids recomputing if it already exists. Returns result in-memory.

### 📌 Code
```python
def get_or_generate_high_quality_annotations(self,
                                             image_ids: List[str],
                                             prompt: str = "individual embryo",
                                             confidence_threshold: float = 0.5,
                                             iou_threshold: float = 0.5,
                                             save_to_self: bool = False) -> Dict:
    result = {}
    existing_hq = self.annotations.get("high_quality_annotations", {})
    
    for exp_id, content in existing_hq.items():
        if (content["prompt"] == prompt and
            content["confidence_threshold"] == confidence_threshold and
            content["iou_threshold"] == iou_threshold):
            filtered = {img_id: ann for img_id, ann in content.get("filtered", {}).items() if img_id in image_ids}
            result.update(filtered)
    
    missing_image_ids = [img_id for img_id in image_ids if img_id not in result]
    
    if missing_image_ids:
        gen = self.generate_high_quality_annotations(
            image_ids=missing_image_ids,
            prompt=prompt,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            overwrite=False,
            save_to_self=save_to_self
        )
        new_filtered = {img_id: ann for img_id, ann in gen.get("filtered", {}).items()}
        result.update(new_filtered)
    
    return result
```

---

## 3. 🔄 Add Method: `generate_missing_high_quality_annotations`

### 📍 Similar to `process_missing_annotations(...)`

### 📌 Code
```python
def generate_missing_high_quality_annotations(self,
                                              prompt: str = "individual embryo",
                                              confidence_threshold: float = 0.5,
                                              iou_threshold: float = 0.5) -> None:
    image_to_exp = self._get_image_to_experiment_map()
    processed_ids = {
        img_id
        for content in self.annotations.get("high_quality_annotations", {}).values()
        for img_id in content.get("filtered", {})
    }

    all_image_ids = list(self.annotations.get("images", {}).keys())
    unprocessed = [img_id for img_id in all_image_ids if img_id not in processed_ids]

    if unprocessed:
        self.generate_high_quality_annotations(
            image_ids=unprocessed,
            prompt=prompt,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            overwrite=True,
            save_to_self=True
        )

    self._unsaved_changes = True
```

---

## 4. 💾 Add Method: `export_high_quality_annotations`

```python
def export_high_quality_annotations(self, export_path: Union[str, Path]) -> None:
    export_path = Path(export_path)
    hq_data = self.annotations.get("high_quality_annotations", {})
    with open(export_path, 'w') as f:
        json.dump(hq_data, f, indent=2)
    if self.verbose:
        print(f"✅ Exported high-quality annotations to: {export_path}")
```

---

## 5. 📥 Add Method: `import_high_quality_annotations`

```python
def import_high_quality_annotations(self, import_path: Union[str, Path], overwrite: bool = False) -> None:
    import_path = Path(import_path)
    with open(import_path, 'r') as f:
        imported = json.load(f)

    self.annotations.setdefault("high_quality_annotations", {})
    for exp_id, content in imported.items():
        if not overwrite and exp_id in self.annotations["high_quality_annotations"]:
            if self.verbose:
                print(f"🔁 Skipped {exp_id} — already exists. Use overwrite=True to replace.")
            continue
        self.annotations["high_quality_annotations"][exp_id] = content
        if self.verbose:
            print(f"✅ Imported annotations for experiment {exp_id}")

    self._unsaved_changes = True
```

---

## 6. ✅ Optional: Add helper `has_high_quality(image_id, prompt)`

```python
def has_high_quality(self, image_id: str, prompt: str = "individual embryo") -> bool:
    for exp_data in self.annotations.get("high_quality_annotations", {}).values():
        if exp_data["prompt"] == prompt and image_id in exp_data.get("filtered", {}):
            return True
    return False
```

---

## 7. 🔁 Update `03_initial_gdino_detections.py` to use the new method

```python
annotations.generate_missing_high_quality_annotations(
    prompt="individual embryo",
    confidence_threshold=args.confidence_threshold,
    iou_threshold=args.iou_threshold
)
annotations.save()
```

---

# ✅ Final Notes

- `generate_high_quality_annotations` now requires `image_ids`, ensuring explicit usage
- The sweep functionality is properly handled by `generate_missing_high_quality_annotations`
- These changes ensure reproducibility, modularity, and avoid unexpected full-dataset filtering

---

# 🧪 QA Checklist for Agent
- [ ] Can `generate_high_quality_annotations()` enforce required image_ids?
- [ ] Does `get_or_generate_high_quality_annotations()` return merged results per-image?
- [ ] Does `generate_missing_high_quality_annotations()` sweep only unprocessed image_ids?
"""