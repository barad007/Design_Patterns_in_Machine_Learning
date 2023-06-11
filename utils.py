import json
from typing import Dict



{
  "samples": [
    {
      "id": 0,
      "path": "/data/content/I1649671381886.jpg"
    },
    {
      "id": 1,
      "path": "/data/content/I1649671382699.jpg"
    }
    ]
}

def save_results(output_json_file: str, id_list: list, path_list: list, pvals: list, pcls: list)-> None:
    """
    A function that saves the evaluation results of all images to a JSON file.

    Parameters
    ----------
    output_json_file : JSON output file path.
    id_list : A list storing the id of an image.
    path_list : A list storing the path to a photo.
    pvals : A list storing information about the probability of assignment to a class.
    pcls : A list storing information about the assigned class.
    """
    try:
        if len(id_list) == len(path_list) == len(pvals) == len(pcls):
            result = []
            for i, idx in enumerate(id_list):
                result.append(
                    {
                        "id": idx,
                        "path": path_list[i],
                        "classPred": pcls[i],
                        "classScore": pvals[i]
                    },
                )
        result = {"preds": result}
        save_report(result=result, path_result_file=output_json_file)
        print(f'Analysis of {len(id_list)} files completed. \nResult file: {output_json_file}')
    except:
        print("Failed to save results.")


def save_report(result: Dict, path_result_file: str) -> None:
    with open(path_result_file, "w") as write_file:
        json.dump(result, write_file)