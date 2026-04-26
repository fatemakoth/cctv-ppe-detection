# ── Paths ─────────────────────────────────────────────────────────────────────
POSE_MODEL       = "yolov8n-pose.pt"
PPE_MODEL_PATH   = "ppe_model/ppe_best.pt"
SNAPSHOT_DIR     = "snapshots"
OUTPUT_DIR       = "checked_videos"      # video_check.py annotated output
CREDENTIALS_FILE = "credentials.json"   # Google Sheets service account key
SHEET_NAME       = "PPE Incidents"      # name of the Google Sheet to log into

# ── Person detection ──────────────────────────────────────────────────────────
PERSON_CONF    = 0.50   # minimum confidence to accept a person detection
MIN_BOX_HEIGHT = 80     # px — boxes shorter than this are skipped (too far / partial)
MIN_ASPECT     = 0.6    # height/width — filters out horizontal / lying-down detections

# ── PPE detection confidence ──────────────────────────────────────────────────
HELMET_OK_CONF      = 0.70   # confidence needed to confirm helmet present
HELMET_NO_CONF      = 0.30   # confidence needed to confirm helmet absent (front-facing)
HELMET_NO_CONF_REAR = 0.55   # stricter absent threshold when person faces away
VEST_OK_CONF        = 0.65
VEST_NO_CONF        = 0.30
VEST_NO_CONF_REAR   = 0.50

# ── PPE class names — must exactly match your model's output labels ────────────
HAS_HELMET = "Hardhat"
NO_HELMET  = "NO-Hardhat"
HAS_VEST   = "Safety Vest"
NO_VEST    = "NO-Safety Vest"

# ── PPE crop regions (fraction of person bounding-box height) ─────────────────
HEAD_TOP  = 0.00;  HEAD_BOT  = 0.30   # head region for helmet check
TORSO_TOP = 0.15;  TORSO_BOT = 0.70   # torso region for vest check

# ── Temporal smoothing ────────────────────────────────────────────────────────
SMOOTH_FRAMES  = 10   # rolling window size (frames)
OK_THRESH      = 6    # votes in window needed to confirm OK
MISSING_THRESH = 4    # votes in window needed to confirm MISSING

# ── Tracking ──────────────────────────────────────────────────────────────────
MERGE_IOU_THRESH = 0.60   # IoU above this → duplicate box, drop lower-confidence one
IOU_MATCH        = 0.25   # used by video_check.py tracker for frame-to-frame matching

# ── Feet-off-floor ────────────────────────────────────────────────────────────
ANKLE_BOX_THRESH  = 0.07   # ankle > 7% of box height above box bottom → feet raised
OFF_FLOOR_THRESH  = 5      # frames in window needed to declare off-floor
FEET_MIN_DURATION = 5.0    # seconds feet must be continuously raised before alerting
FOOT_CONF_MIN     = 0.40   # minimum keypoint confidence to trust ankle position
MIN_ANKLES        = 1      # need at least this many visible ankles

# ── Rear-facing detection ─────────────────────────────────────────────────────
REAR_CONF_MIN = 0.30   # nose keypoint confidence below this → facing away

# ── Occlusion detection ───────────────────────────────────────────────────────
SHOULDER_CONF_MIN = 0.40   # below this → shoulder not visible
EDGE_MARGIN       = 10     # px — box within this distance of frame edge → clipped

# ── Incident logging ──────────────────────────────────────────────────────────
COOLDOWN_SEC = 15   # seconds between logging the same violation for the same person

# ── Snapshots ─────────────────────────────────────────────────────────────────
MAX_SNAPSHOTS = 500    # oldest files are deleted once this count is exceeded
MIN_FREE_GB   = 0.5    # stop saving snapshots if free disk drops below this (GB)
JPEG_QUALITY  = 70     # JPEG compression — 70 is ~3-4x smaller than default, still clear

# ── RTSP reconnect ────────────────────────────────────────────────────────────
RECONNECT_DELAYS         = [2, 5, 10, 30]   # wait seconds between reconnect attempts
RECONNECT_FAIL_THRESHOLD = 10               # consecutive bad reads before reconnecting

# ── Keypoint indices (YOLOv8-pose, COCO layout — do not change) ──────────────
NOSE_KP        = 0
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16
