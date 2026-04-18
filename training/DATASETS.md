# Datasets & repos for US-style turn / U-turn / stop detection (YOLO / Ultralytics)

This document answers: *where to get bounding-box data*, *what classes exist*, *licensing*, and *how it plugs into this repo*.

---

## Requirements recap

| Need | Object detection (bbox) | Compatible with Ultralytics YOLOv8 |
|------|-------------------------|-------------------------------------|
| Stop sign | Yes | Yes |
| Left turn only | Yes | Yes (if labels exist) |
| Right turn only | Yes | Yes |
| U-turn (permitted lane) | Often **missing** | Yes (if you add images) |

**Reality:** Almost no public set uses *exactly* your PDF wording. You will **merge** sources and/or **add 100–300 photos** of your printed signs (especially **U-turn permitted**, R3-19a).

---

## Candidate comparison

### 1) **LISA Traffic Sign Dataset** (US, bbox in CSV)

| | |
|--|--|
| **What** | ~6.6k frames, ~7.8k sign instances, **47 US sign types**, CSV annotations with box coordinates. |
| **Bbox detection** | Yes (rectangles in CSV). |
| **US-specific** | Yes. |
| **Turn / U-turn** | LISA includes many regulatory/warning types; **exact** “left turn only” / “U-turn only” strings depend on their taxonomy — you **list unique `Tag` values** after download and map with `training/lisa_class_map.example.json`. |
| **Format** | CSV (`allAnnotations.csv`), not YOLO — **use `training/scripts/lisa_csv_to_yolo.py`**. |
| **License** | Academic / research use; **read UCSD / dataset page** before commercial rover use. |
| **Links** | [UCSD LISA](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html), [Kaggle mirror](https://www.kaggle.com/datasets/omkarnadkarni/lisa-traffic-sign), [AAU portal](https://vbn.aau.dk/en/datasets/lisa-traffic-sign-dataset) |
| **Integration** | Medium (conversion script in this repo). **Best US-native starting point.** |

---

### 2) **Roboflow Universe** (many projects, YOLO export)

| | |
|--|--|
| **What** | Curated projects with **YOLOv8** export (zip contains `data.yaml`, `train/images`, `train/labels`). |
| **Bbox detection** | Yes. |
| **US-specific** | Varies by project; some global / mixed. |
| **Turn / U-turn** | Several projects list `left_turn`, `right_turn`, `no_left_turn`, etc. **Verify class list** on the project page before training. |
| **License** | **Per dataset** (often CC BY 4.0 on Universe; check each card). |
| **Links** | [US Road Signs](https://universe.roboflow.com/us-traffic-signs-pwkzx/us-road-signs) · [Traffic Sign Detection (102 classes)](https://universe.roboflow.com/traffic-sign-detection-80yhm/traffic-sign-detection-fvaoc) · [traffic-sign-detection-yolov8](https://universe.roboflow.com/university-km5u7/traffic-sign-detection-yolov8-awuus) |
| **Integration** | **Easiest** — unzip, point `data.yaml` at `yolo train`. Use `training/scripts/fetch_roboflow_dataset.py` if you use a Roboflow API key. |

---

### 3) **Mapillary Traffic Sign Dataset**

| | |
|--|--|
| **What** | Very large, **300+ sign categories**, bbox annotations, global imagery. |
| **Bbox detection** | Yes. |
| **US-specific** | No — **not US-only**; you filter/map Mapillary class IDs to your rover taxonomy. |
| **License** | **CC BY-NC-SA** — **non-commercial**; check if your pledge/commercial use is allowed. |
| **Links** | [Mapillary dataset page](https://www.mapillary.com/dataset/trafficsign), [paper (arXiv)](https://arxiv.org/abs/1909.04422) |
| **Integration** | Harder (taxonomy mapping + NC license). **Strong for research, not default for a quick rover.** |

---

### 4) **German / European benchmarks (GTSDB, DFG, etc.)**

| | |
|--|--|
| **What** | Many **arrow / mandatory** signs; good **pretrain** for *shape* of arrows. |
| **US match** | Sign shapes differ from MUTCD; use as **auxiliary pretrain** or **domain adaptation**, not as final US labels without relabeling. |
| **Links** | [GTSDB](https://benchmark.ini.rub.de/gtsdb_dataset.html), [DFG dataset](https://www.vicos.si/resources/dfg/) |

---

## Recommended **single best path** for *this* repo

**Combine:**

1. **Primary data (US, bbox):** **LISA** → convert with **`training/scripts/lisa_csv_to_yolo.py`** + **`training/lisa_class_map.example.json`** to collapse LISA tags into your 5 rover classes where possible.
2. **Extra diversity & turn arrows:** Download one **Roboflow** detection project that explicitly lists **left/right turn** classes → export **YOLOv8**, merge labels (script or manual union of `names` in YAML).
3. **U-turn permitted (R3-19a):** Often **underrepresented**. **Mandatory:** photograph your **`printable_signs/MUTCD_R3-19a_u_turn_only.svg`** (and real roads if possible), label in **Roboflow / CVAT / Label Studio**, export YOLO, **merge** into the same dataset.

**Why not only Roboflow?** Variable quality and class naming; **LISA** anchors **US** appearance.  
**Why not only LISA?** Instance count per rare class can be low; Roboflow augments diversity.  
**Why not Mapillary as default?** **Non-commercial** license + heavy class mapping.

---

## Class remapping (manual step you must do once)

1. Run on LISA CSV:  
   `python training/scripts/lisa_csv_to_yolo.py --lisa-root ... --list-tags`  
   → prints all `Tag` strings.
2. Edit **`training/lisa_class_map.json`** (copy from example): map each relevant LISA tag → one of:  
   `stop sign` | `left turn only` | `right turn only` | `u turn only` | `traffic light`
3. Tags you **cannot** map → drop or map to closest (document in JSON comments — JSON has no comments, use sidecar `lisa_class_map.README.txt` if needed).

For Roboflow exports, open their `data.yaml` `names:` and **rename** to match **`robot/sign_policy.py`** (`DEFAULT_LABEL_TO_ACTION` keys) or extend `sign_policy.py` to match Roboflow.

---

## Risks & gaps

| Risk | Mitigation |
|------|------------|
| No “U-turn only” in LISA | Add **your own** labeled images of R3-19a. |
| Class name mismatch | Align `data.yaml` `names` ↔ `sign_policy.py`. |
| `coco_rover` filter hides new classes | Inference: **`--class-filter none`**. |
| License (LISA / Mapillary) | Read terms before public demo or product. |

---

## Repo integration map (where things live)

| Concern | Location |
|---------|----------|
| Train CLI | `python -m robot.train_yolo --data <yaml> ...` → `robot/train_yolo.py` |
| Dataset YAML | `training/rover_finetune.yaml` (template) or Roboflow’s `data.yaml` |
| Weights path | `robot/run_robot.py` `--weights`, `robot/camera_test.py` `--yolo` |
| Live labels → motors | `robot/sign_policy.py` `DEFAULT_LABEL_TO_ACTION` |
| COCO clutter filter | `robot/detection_filter.py` — use **`none`** for custom weights |
| Printable targets | `printable_signs/`, `training/CV_Rover_Signs.pdf` |

---

## Exact commands (summary)

**Train (after `data.yaml` is ready):**

```bash
python -m robot.train_yolo --data training/rover_finetune.yaml --model yolov8n.pt --epochs 100 --batch 8
```

**Inference / test:**

```bash
python -m robot.camera_test --yolo runs/detect/train/weights/best.pt --class-filter none --conf 0.25
python -m robot.run_robot --weights runs/detect/train/weights/best.pt --class-filter none --display
```

See also **`training/ROVER_SIGNS.txt`** and **`training/rover_signs.yaml.example`**.
