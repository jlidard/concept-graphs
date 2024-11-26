
import json
import pickle
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_file", type=str, default=None)
    parser.add_argument("--scene_name", type=str, default='scene')
    return parser



def main(args):

    assert not (args.edge_file is None), \
    "edge_file must be provided."

    object_dict = {}

    if args.edge_file is not None:
        # Load edge file for the scene graph
        with open(args.edge_file, "r") as f:
            edges = json.load(f)

        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']

            object1 = edge["object1"]
            object2 = edge["object2"]

            if object1['id'] not in object_dict:
                vertex1 = {}
                vertex1['semantic_label'] = object1['object_tag']
                vertex1['centroid'] = object1['bbox_center']
                vertex1['bounding_box'] = object1['bbox_extent']
                object_dict[object1['id']] = vertex1

            if object2['id'] not in object_dict:
                vertex2 = {}
                vertex2['semantic_label'] = object2['object_tag']
                vertex2['centroid'] = object2['bbox_center']
                vertex2['bounding_box'] = object2['bbox_extent']
                object_dict[object2['id']] = vertex2

        
        print(object_dict)

        with open (f"/home/pbrick/concept-graphs/conceptgraph/objectvertices/{args.scene_name}_vertices.json", "w") as f:
            json.dump(object_dict, f, indent=4)

        with open (f"/home/pbrick/concept-graphs/conceptgraph/objectvertices/{args.scene_name}_vertices.pkl", "wb") as f:
            pickle.dump(object_dict, f)



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)