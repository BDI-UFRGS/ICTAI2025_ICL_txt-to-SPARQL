import json
from experiment import Experiment
from datetime import datetime

def generate_report(experiment: Experiment):
    
    metadata = {
        "description": experiment.description,
        "model": experiment.model.value,
        "rounds_per_query": len(experiment.rounds),
        "date": datetime.now().strftime("%Y-%b-%d"),
        "macro f1-score": {
            "Round 0": round(experiment.rounds[0].macro_f1, 3),
            "Round 1": round(experiment.rounds[1].macro_f1, 3),
            "Round 2": round(experiment.rounds[2].macro_f1, 3),
            "Round 3": round(experiment.rounds[3].macro_f1, 3),
            "Round 4": round(experiment.rounds[4].macro_f1, 3),
            "Average": round(sum([experiment.rounds[i].macro_f1 for i in range(len(experiment.rounds))]) / len(experiment.rounds), 4)
        },
        "semantic accuracy": {
            "Round 0": round(experiment.rounds[0].semantic_accuracy, 3),
            "Round 1": round(experiment.rounds[1].semantic_accuracy, 3),
            "Round 2": round(experiment.rounds[2].semantic_accuracy, 3),
            "Round 3": round(experiment.rounds[3].semantic_accuracy, 3),
            "Round 4": round(experiment.rounds[4].semantic_accuracy, 3),
            "Average": round(sum([experiment.rounds[i].semantic_accuracy for i in range(len(experiment.rounds))]) / len(experiment.rounds), 4)
        }
    }

    all_data = []

    for execution_index in range(len(experiment.rounds[0].executions)):

        example_queries = []

        for example_query in experiment.rounds[0].executions[execution_index].example_queries:
            example_query_ = {
                "NL Query": example_query['example'].nl_query.nl_query,
                "SparQL Query": example_query['example'].sparql_query,
                "Cosine Similarity": round(float(example_query['cosine_similarity']), 3)
            }

            if 'cluster' in example_query:
                example_query_["Cluster"] = int(example_query['cluster'])

            example_queries.append(example_query_)

        dados = {
            "Gold Query": {
                "NL Query": experiment.rounds[0].executions[execution_index].gold_query.nl_query.nl_query,
                "SparQL Query": experiment.rounds[0].executions[execution_index].gold_query.sparql_query,
                "Endpoint Output": list(experiment.rounds[0].executions[execution_index].gold_query.endpoint_output) if isinstance(experiment.rounds[0].executions[execution_index].gold_query.endpoint_output, set) else experiment.rounds[0].executions[execution_index].gold_query.endpoint_output
            },  
            "Generated Queries": [
                {
                    "Round": round_index,
                    "SparQL Query": experiment.rounds[round_index].executions[execution_index].generated_query.sparql_query,
                    "Endpoint Output": list(experiment.rounds[round_index].executions[execution_index].generated_query.endpoint_output) if isinstance(experiment.rounds[round_index].executions[execution_index].generated_query.endpoint_output, set) else experiment.rounds[round_index].executions[execution_index].generated_query.endpoint_output,
                    "Is Semantic Equivalent": experiment.rounds[round_index].executions[execution_index].generated_query.is_equivalent,
                    "F1-Score": float(
                        experiment.rounds[round_index].executions[execution_index].generated_query.confusion_matrix.f1_score
                        if experiment.rounds[round_index].executions[execution_index].generated_query.confusion_matrix is not None
                        else 0
                    ),
                    "Hallucinated URIs": list(experiment.rounds[round_index].executions[execution_index].generated_query.hallucinated_uris) if experiment.rounds[round_index].executions[execution_index].generated_query.hallucinated_uris is not None else None,
                    "Has Triple Flip": experiment.rounds[round_index].executions[execution_index].generated_query.has_triple_flip,
                    "Has SparQL Parsing Error": experiment.rounds[round_index].executions[execution_index].generated_query.has_query_parsing_error
                }
            for round_index in range(len(experiment.rounds))
            ],
            "Example Queries": example_queries
        }
        all_data.append(dados)

    final_data = {
        "metadata": metadata,
        "data": all_data
    }

    with open(f'../data/reports/{experiment.id}.json', "w", encoding="utf-8") as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)