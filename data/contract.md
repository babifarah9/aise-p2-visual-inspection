# Data Contract: Camera → Inspection Model

| Field | Type | Rule |
| --- | --- | --- |
| `image` | 16 × 16 floats in [0, 1] (production: camera native, resized) | Exactly SIZE rows of SIZE values |
| `part_id` | string | Unique per part |
| `camera_id` | string | One of the cell's four cameras |
| `captured_at` | ISO 8601 timestamp | Monotonic per camera |
| `label` | `ok`, `scratch`, `pit` | Only on audited parts; set by a certified inspector |

Raw images are never committed here.
