# Update the manualQC.txt file with an agent prompt at the top
content = """# Agent Prompt

You are an autonomous AI assistant tasked with implementing and maintaining the manual QC pipeline for MorphSeq image data. Your responsibilities include:


# image_qaulituy_qc overview

1. Purpose
Provide a simple, scalable pipeline for recording manual or automatically generated QC flags on MorphSeq images. this is done prior to grounded_sam embryo detection and segmentation 


scripts neccesary to generate (and maintain) qc 
- **image_quality_qc.csv**: Records images that did not pass QC and did stored in data folder
  -    - Ensure the required columns (`experiment_id`, `video_id`, `image_id`, `qc_flag`, `notes`, `annotator`) are present.
- image_quality_qc_utils.py: (functions for applying qc manually or automatically) stored with utils 
_ 02_image_quality_qc.py: script that mainly applies to new files in the embryo_metadata_json by cross referecning image_qualitt_qc entried with the metadata embryo frm previous step. however there should be functionallity to process individual experiment_ids, video_ids, and image_id (note that it will all be convertied to lists of corresponding image_ids)

Note: ther 

There will be 

image_quality quality control which happens before segmentation grounded_sam pipeline 

there need to be scripts capable of performing manual qc and automatied qc designations

there will be a .csv file that will have a annotator "nlammers" "mcolon" or if automatic "auto" 

if an image_id is in this file then it has a image_quality qc flag. 

qc_flag_type column should have one of the stored (in the image_qc_utils.py file) 

NOTE these scripts should be simple! do not OVERCOMPLICATE THis . 

after

---

1. File layout

\`\`\`
morphseq/
├── data/
│   └── qality_control/
│       ├── manual_qc.csv             # CSV columns: experiment_id, video_id, image_id, qc_flag, notes, annotator
│       └── comprehensive_qc.csv      # Generated as needed, includes all images (PASS/FAIL)
│
├── morphseq/
│   ├── QCUtils.py                    # Core routines for manual QC (pseudocode below) and automated 
│   └── ComprehensiveQCUtils.py       # Separate routines for generating comprehensive QC
\`\`\`

---

1. Core routines (QCUtils.py)

FUNCTION load_manual_qc()
  → read \`data/qc/manual_qc.csv\` into a table
  → return table

FUNCTION save_manual_qc(table)
  → overwrite \`data/qc/manual_qc.csv\` with table

FUNCTION parse_image_id(image_id)
  # \`image_id\` format: "YYYY-MM-DD_<well>_<frame>"
  → split on "_" → [experiment_id, well, frame]
  → video_id = experiment_id + "_" + well
  → return (experiment_id, video_id, frame)
  (however this should just be extracted from the experimnent_metadatajson)

FUNCTION flag_qc(
    EITHER image_ids[] 
      OR (video_id, frames[])
  , qc_flag       # "PASS" or "FAIL"
  , annotator     # REQUIRED: string identifying who flags ("auto" if automatically generated)
  , notes = ""    # free text
  , overwrite = FALSE
  )
  → IF annotator is empty:
      ERROR "Annotator must be provided"
  → table = load_manual_qc()

  → BUILD list \`to_process\` of (exp, vid, img):
    IF image_ids given:
      FOR img in image_ids:
        exp, vid, frame = parse_image_id(img)
        ADD (exp, vid, img)
    ELSE IF video_id + frames given:
      exp = text before first "_" in video_id
      FOR frame in frames:
        img = video_id + "_" + frame
        ADD (exp, video_id, img)
    ELSE:
      ERROR "Must pass image_ids OR video_id + frames"

  → FOR each (exp, vid, img) in to_process:
      mask = rows matching (experiment_id=exp, video_id=vid, image_id=img)
      IF mask exists:
        IF NOT overwrite:
          ERROR "Record exists for {img}; use overwrite=TRUE to replace."
        ELSE:
          UPDATE qc_flag, notes, annotator in those rows
      ELSE:
        APPEND new row (exp, vid, img, qc_flag, notes, annotator)

  → save_manual_qc(table)

FUNCTION remove_qc(
    EITHER image_ids[] 
      OR (video_id, frames[])
  , annotator     # REQUIRED: string identifying who removes
  , overwrite = FALSE
  )
  → IF annotator is empty:
      ERROR "Annotator must be provided"
  → table = load_manual_qc()
  → BUILD list \`to_remove\` same as in flag_qc

  → FOR each (exp, vid, img) in to_remove:
      mask = rows matching (experiment_id=exp, video_id=vid, image_id=img)
      IF NO mask AND NOT overwrite:
        ERROR "No QC exists for {img}"
      ELSE:
        DROP those rows

  → save_manual_qc(table)

---

4. Comprehensive QC routines (ComprehensiveQCUtils.py)

FUNCTION load_experiment_data()
  → read \`experimentmetadatadata.json\` into a data structure
  → return experiment data

FUNCTION generate_comprehensive_qc()
  → load experiment data (all image IDs)
  → load image_quality_qc.csv (failed images)
  → CREATE comprehensive_qc.csv with all image IDs
      - Mark images in manual_qc.csv as FAIL with corresponding notes/annotator
      - Mark all other images as PASS
  → save comprehensiv_image_qc.csv in sae location as other file image_qc.csv

---

1. User-facing script

SCRIPT manualimage_quaitu_QC:
  INPUT:
    • mode            # "flag" or "remove"
    • image_ids[]     # OR
    • video_id + frames[]
    • qc_flag         # required if mode = "flag"
    • annotator       # required for both flag & remove
    • notes           # optional (only for flag)
    • overwrite?      # default false

  IF mode == "flag":
    CALL flag_qc(…)
  ELSE IF mode == "remove":
    CALL remove_qc(…)
  ELSE:
    ERROR "Unknown mode"

  PRINT "manual_qc.csv updated."
Note: 
should have auto_qc and manual_qc image_quality functions as wrappers to flag_image_quality_qc since they are pushing to the same file... 
---

6. Next steps

1. Implement functions in \`
image_qualitty_qx_utils.py\`:
   - \`load_manual_qc\` → \`flag_qc\` → \`remove_qc\` → script logic.
1. Add tests to ensure:
   - All images have a QC entry in comprehensive file.
   - \`annotator\` requirement is enforced.
   - Overwrite behavior and error conditions.
2. Later, extend \`QCUtils.py\` with QC: auto-detect blur, fog, focus metrics.( these will be determined automatically (or  manually which is why there is a shared standardized list in the image_qualitt_qc_utils.py file  ) )


also there should be a check in the flag_qc functions to if there already is qc iamge_ids in the function 
---


Suggestions & Best Practices

- **Annotator column**: Enforces accountability; make it required.
- **Version control**: Pushing \`manual_qc.csv\` to Git is fine if it's under a few MB. For larger scales, consider DVC or Git LFS.
- **Schema validation**: Add a quick CSV‐schema check on load (columns and types).
- **Testing**: Mock \`experimentdata.json\` to generate initial comprehensive CSV.
- **Consistent data**: Use the JSON source to initialize the comprehensive CSV so every image is represented.
- **Documentation**: Keep this spec aligned with code comments and update when adding flags or QC types.
"""

file_path = "/mnt/data/manualQC.txt"
with open(file_path, "w") as f:
    f.write(content)


FINAL and important notes 

 this 


 4. **Best Practices & Safety**  
   - Perform atomic writes to avoid data corruption.  
   - Provide clear, user-friendly error messages.  
   - Suggest optimizations or report potential inconsistencies.  
   - Keep documentation (this spec) synced with code changes.  
   - Consider version control, backups, and scalability as data grows.


sorry i dont want a separet manuak_qc file,
all image_qualitt_qcs should go to a shared .csv where a annotator colum keeps track if it was manually or automatically along with the actual qc flag