import sys
from transformers import pipeline
from typing import Any
# sys.path.insert(0,"baseline")
# from bart import output, gold_metareview

# TODO: add the dependency to requirements.txt

class Evaluator:
    def __init__(self, predictions:list, references:list):
        self.predictions = predictions
        self.references = references

    def _rouge(self) -> list[float]:
        from evaluate import load
        rouge=load('rouge')
        return rouge.compute(predictions=self.predictions,
                         references=self.references,
                         rouge_types=['rougeL'],
                         use_aggregator=False)['rougeL']
    
    def _bertscore(self, model_type:str) -> dict[str, list[float]]:
        from evaluate import load
        bertscore = load("bertscore")
        result= bertscore.compute(predictions=self.predictions,
                                references=self.references,
                                model_type=model_type)
        
        del result['hashcode']
        return result
    
    def _factCC(self, reviews:list[list[str]], meta_reviews:list[str]) -> list[dict[str, Any]]:
        '''
        https://huggingface.co/manueldeprada/FactCC
        Note: FactCC is not comparing reference and predictions but summaries and source texts
        
        Return:
        >>> output [{'label': 'INCORRECT', 'score': 0.9979124665260315}, {'label': 'CORRECT', 'score': 0.879124665260315}, ...]
        '''
       
        pipe=pipeline(model="manueldeprada/FactCC")
        concatenated_reviews = [' '.join(review_list) for review_list in reviews]
    
        data = [[[concatenated_reviews[i], meta_reviews[i]]] for i in range(len(concatenated_reviews))]
        return pipe(data, truncation='only_first',padding='max_length')


    def _summaC(self, reviews:list[list[str]], meta_reviews:list[str]) -> list[float]:
        # TODO: SummaC:https://github.com/tingofurro/summac
        # model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
        from summac.model_summac import SummaCConv

        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")
        concatenated_reviews = [' '.join(review_list) for review_list in reviews]

        scores_conv = []
        data = [(concatenated_reviews[i], meta_reviews[i]) for i in range(len(concatenated_reviews))]
        for doc, summary in data:
            # scores_zs.append(model_zs.score([doc], [summary])["scores"][0].item())
            scores_conv.append(model_conv.score([doc], [summary])["scores"][0])
        
        return scores_conv
    
    def _discoScore(self, meta_reviews:list[str], reviews:list[list[str]]) -> list[dict[str, float]]:
        # TODO: https://github.com/AIPHES/DiscoScore
        # TODO: which discourse metric to use?
        # TODO: not sure what the system means
        from disco_score import DiscoScorer
        disco_scorer = DiscoScorer(device='cpu', model_name='bert-base-uncased')

        result = []
        for s, refs in zip(meta_reviews, reviews):
            s = s.lower()
            refs = [r.lower() for r in refs]
            result.append({"EntityGraph": disco_scorer.EntityGraph(s, refs),
                           "LexicalChain": disco_scorer.LexicalChain(s, refs),
                           "RC": disco_scorer.RC(s, refs),
                           "LC": disco_scorer.LC(s, refs)})
                        #    "DS_Focus_NN": disco_scorer.DS_Focus_NN(s, refs),# FocusDiff 
                        #    "DS_SENT_NN":disco_scorer.DS_SENT_NN(s, refs)}) # SentGraph
        return result

    def evaluate(self, metric, **kwargs):
        if metric == "rouge_L":
            return self._rouge()
        elif metric == "bertscore":
            return self._bertscore(model_type=kwargs.get("model_type", "distilbert-base-uncased"))
        elif metric == "factCC":
            return self._factCC(
                reviews=kwargs["reviews"],
                meta_reviews=kwargs["meta_reviews"],
            )
        elif metric == "summaC":
            return self._summaC(
                reviews=kwargs["reviews"],
                meta_reviews=kwargs["meta_reviews"],
            )
        elif metric == "disco":
            return self._discoScore(
                reviews=kwargs["reviews"],
                meta_reviews=kwargs["meta_reviews"],
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")