# reference scores for all dynamics shift tasks

REF_MIN_SCORE = {
    'halfcheetah' : -280.178953 ,
    'halfcheetah-morph' : -280.178953 ,
    'halfcheetah-gravity' : -280.178953 ,
    'hopper' : -26.3360015397715 ,
    'hopper-morph' : -26.3360015397715 ,
    'hopper-gravity' : -26.3360015397715 ,
    'walker2d' : 10.079455055289959 ,
    'walker2d-morph' : 10.079455055289959 ,
    'walker2d-gravity' : 10.079455055289959 ,
    'ant' : -325.6 ,
    'ant-morph' : -325.6 ,
    'ant-gravity' : -325.6 ,
}

REF_MAX_SCORE = {
    'halfcheetah' : 7065.03 ,
    'halfcheetah-morph' : 9713.59 ,
    'halfcheetah-gravity' : 9509.15 ,
    'hopper' : 2842.73 ,
    'hopper-morph' : 3152.75 ,
    'hopper-gravity' : 3234.3 ,
    'walker2d' : 3257.51 ,
    'walker2d-morph' : 4398.43 ,
    'walker2d-gravity' : 5194.713 ,
    'ant' : 5122.57 ,
    'ant-morph' : 5722.01 ,
    'ant-gravity' : 4317.065 ,
}

def get_normalized_score(env_name, score):
    ref_min_score = REF_MIN_SCORE[env_name]
    ref_max_score = REF_MAX_SCORE[env_name]
    return (score - ref_min_score) / (ref_max_score - ref_min_score) * 100