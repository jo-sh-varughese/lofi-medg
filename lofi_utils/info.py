from collections import OrderedDict

PADCHEST_GR_LABEL_DICT = OrderedDict([
    ("negative", ["N/A"]),  # TODO:

    ("aortic elongation", [
        "aortic elongation",
        "aortic button enlargement",
        "descendent aortic elongation",
        "supra aortic elongation",
        "ascendent aortic elongation"
    ]),

    ("cardiomegaly", [
        "cardiomegaly",
        "pericardial effusion"
    ]),

    ("nodule", [
        "pseudonodule",
        "calcified granuloma",
        "nodule",
        "nipple shadow",
        "pulmonary mass",
        "granuloma",
        "multiple nodules",
        "end on vessel",
        "mass",
        "soft tissue mass",
        "miliary opacities",
        "pleural mass"
    ]),

    ("pleural effusion", [
        "pleural effusion",
        "costophrenic angle blunting",
        "minor fissure thickening",
        "loculated pleural effusion",
        "loculated fissural effusion",
        "fissure thickening",
        "major fissure thickening"
    ]),

    ("scoliosis", [
        "scoliosis"
    ]),

    ("vertebral degenerative changes", [
        "vertebral degenerative changes",
        "vertebral anterior compression",
        "vertebral compression"
    ]),

    ("hyperinflated lung", [
        "air trapping",
        "flattened diaphragm",
        "hyperinflated lung"
    ]),

    ("vascular hilar enlargement", [
        "vascular hilar enlargement",
        "hilar congestion",
        "pulmonary artery enlargement"
    ]),

    ("atelectasis", [
        "laminar atelectasis",
        "atelectasis",
        "lobar atelectasis",
        "atelectasis basal",
        "segmental atelectasis",
        "total atelectasis"
    ]),

    ("aortic atheromatosis", [
        "aortic atheromatosis"
    ]),

    ("pleural thickening", [
        "apical pleural thickening",
        "pleural thickening",
        "calcified pleural thickening",
        "calcified pleural plaques"
    ]),

    ("interstitial pattern", [
        "interstitial pattern",
        "ground glass pattern",
        "reticular interstitial pattern",
        "reticulonodular interstitial pattern"
    ]),

    ("alveolar pattern", [
        "alveolar pattern",
        "consolidation",
        "cavitation",
        "air bronchogram",
        "abscess"
    ]),

    ("electrical device", [
        "pacemaker",
        "dual chamber device",
        "single chamber device",
        "dai",
        "electrical device"
    ]),

    ("hemidiaphragm elevation", [
        "diaphragmatic eventration",
        "hemidiaphragm elevation"
    ]),

    ("fracture", [
        "callus rib fracture",
        "rib fracture",
        "humeral fracture",
        "clavicle fracture",
        "vertebral fracture",
        "fracture"
    ]),

    ("hypoexpansion", [
        "volume loss",
        "hypoexpansion"
    ]),

    ("central venous catheter", [
        "central venous catheter via jugular vein",
        "central venous catheter via subclavian vein",
        "central venous catheter",
        "reservoir central venous catheter"
    ]),

    ("hiatal hernia", [
        "hiatal hernia"
    ]),

    ("endotracheal tube", [
        "endotracheal tube",
        "tracheostomy tube"
    ]),

    ("nsg tube", [
        "NSG tube"
    ]),

    ("bronchiectasis", [
        "bronchiectasis"
    ]),

    ("goiter", [
        "goiter"
    ]),

    ("osteopenia", [
        "osteopenia",
        "osteoporosis"
    ]),

    ("other entities", [
        "chronic changes",
        "infiltrates",
        "fibrotic band",
        "increased density",
        "kyphosis",
        "sternotomy",
        "suture material",
        "hilar enlargement",
        "metal",
        "gynecomastia",
        "calcified densities",
        "mammary prosthesis",
        "osteosynthesis material",
        "bronchovascular markings",
        "sclerotic bone lesion",
        "tracheal shift",
        "bullas",
        "azygos lobe",
        "mastectomy",
        "superior mediastinal enlargement",
        "mediastinic lipomatosis",
        "mediastinal enlargement",
        "vascular redistribution",
        "axial hyperostosis",
        "surgery breast",
        "non axial articular degenerative changes",
        "surgery",
        "thoracic cage deformation",
        "artificial heart valve",
        "costochondral junction hypertrophy",
        "pneumothorax",
        "surgery neck",
        "calcified adenopathy",
        "adenopathy",
        "mediastinal mass",
        "surgery lung",
        "chest drain tube",
        "obesity",
        "artificial mitral heart valve",
        "central vascular redistribution",
        "pectum excavatum",
        "heart valve calcified",
        "humeral prosthesis",
        "air fluid level",
        "cervical rib",
        "kerley lines",
        "pneumoperitoneo",
        "abnormal foreign body",
        "artificial aortic heart valve",
        "catheter",
        "lytic bone lesion",
        "prosthesis",
        "sternoclavicular junction hypertrophy",
        "subacromial space narrowing",
        "subcutaneous emphysema",
        "surgery heart",
        "aortic aneurysm",
        "aortic endoprosthesis",
        "azygoesophageal recess shift",
        "blastic bone lesion",
        "calcified fibroadenoma",
        "cyst",
        "hydropneumothorax",
        "lung vascular paucity",
        "surgery humeral",
        "Chilaiditi sign",
        "endoprosthesis",
        "gastrostomy tube",
        "mediastinal shift",
        "pectum carinatum",
        "ventriculoperitoneal drain tube"
    ]),
])

PADCHEST_GR_LABEL_DICT_keys = list(PADCHEST_GR_LABEL_DICT.keys())

PADCHEST_GR_LABEL_DICT_inverse = OrderedDict()
for group, findings in PADCHEST_GR_LABEL_DICT.items():
    for f in findings:
        PADCHEST_GR_LABEL_DICT_inverse[f] = group
