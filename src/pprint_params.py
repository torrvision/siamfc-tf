from pprint import pprint


def pprint_params(params_list, group_names):
    print '\n#### SETTINGS'
    for idx, p in enumerate(params_list):
        print '\n## ' + group_names[idx]
        pprint(p)
