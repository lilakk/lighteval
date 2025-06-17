
class F1_score_HotpotQA:
    """
    A class to compute the F1 score as defined in the HotpotQA repository.
    This implementation mirrors the logic in `hotpot_evaluate_v1.py`.
    """

    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
    ):
        """An F1 score class that replicates the HotpotQA evaluation.

        Args:
            aggregation_function (callable, optional): How to aggregate scores if multiple
                predictions or gold standards are provided for a single sample.
                Defaults to max.
        """
        self.aggregation_function = aggregation_function

    @staticmethod
    def _normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets. The HotpotQA dataset has one gold answer per question.
            predictions (list[str]): Predicted strings.

        Returns:
            float: Aggregated score over the current sample's items.
        """
        results = []
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(self, gold: str, pred: str) -> float:
        """
        Compares two strings and computes the F1 score based on the multiset of their tokens,
        following the HotpotQA evaluation script.
        """
        normalized_prediction = self._normalize_answer(pred)
        normalized_gold = self._normalize_answer(gold)

        # Handle yes/no answers, a special case in the official script
        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_gold:
            return 0.0
        if normalized_gold in ["yes", "no", "noanswer"] and normalized_prediction != normalized_gold:
            return 0.0

        prediction_tokens = normalized_prediction.split()
        gold_tokens = normalized_gold.split()

        if not prediction_tokens or not gold_tokens:
            return 0.0 if prediction_tokens != gold_tokens else 1.0

        common = Counter(prediction_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1


f1_score_hotpotqa = SampleLevelMetric(
        metric_name="f1",
        sample_level_fn=F1_score_HotpotQA().compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


class Entailment_fuzzy_HotpotQA:
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
    ):
        """An Entailment fuzzy match class that replicates the HotpotQA evaluation.
        For each word in the gold answer, it computes the fuzzy match score with all words in the generated answer and takes the maximum.
        The final score is the average of these maximum scores.

        Args:
            aggregation_function (callable, optional): How to aggregate scores if multiple
                predictions or gold standards are provided for a single sample.
                Defaults to max.
        """
        self.aggregation_function = aggregation_function

    @staticmethod
    def _normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def _edit_similarity(s1: str, s2: str) -> float:
        """Compute the edit similarity between two strings."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        edist = edit_distance(s1, s2)
        return 1.0 - edist / max(len(s1), len(s2))

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets. The HotpotQA dataset has one gold answer per question.
            predictions (list[str]): Predicted strings.

        Returns:
            float: Aggregated score over the current sample's items.
        """
        results = []
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(self, gold: str, pred: str) -> float:
        """
        For each word in the gold answer, finds the best fuzzy match in the predicted answer.
        The final score is the average of these fuzzy matches.
        """
        normalized_prediction = self._normalize_answer(pred)
        normalized_gold = self._normalize_answer(gold)

        if normalized_gold in ["yes", "no", "noanswer"]:
            return 1.0 if normalized_gold in normalized_prediction else 0.0

        prediction_tokens = normalized_prediction.split()
        gold_tokens = normalized_gold.split()

        if not gold_tokens:
            return 1.0 if not prediction_tokens else 0.0

        if not prediction_tokens:
            return 0.0

        fuzzy_scores = []
        for gold_token in gold_tokens:
            # Find the best match for the current gold token in the prediction tokens
            scores = [self._edit_similarity(gold_token, pred_token) for pred_token in prediction_tokens]
            fuzzy_scores.append(max(scores))

        return np.mean(fuzzy_scores) if fuzzy_scores else 0.0


entailment_fuzzy_hotpotqa = SampleLevelMetric(
    metric_name="entailment_fuzzy",
    sample_level_fn=Entailment_fuzzy_HotpotQA().compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


class Entailment_fuzzy_HotpotQA_v2:
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
    ):
        """An Entailment fuzzy match class that replicates the HotpotQA evaluation.
        For each word in the gold answer, it computes the fuzzy match score with all words in the generated answer and takes the maximum.
        The final score is the average of these maximum scores.

        Args:
            aggregation_function (callable, optional): How to aggregate scores if multiple
                predictions or gold standards are provided for a single sample.
                Defaults to max.
        """
        self.aggregation_function = aggregation_function

    @staticmethod
    def _normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def _edit_similarity(s1: str, s2: str) -> float:
        """Compute the edit similarity between two strings."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        edist = edit_distance(s1, s2)
        return 1.0 - edist / max(len(s1), len(s2))

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets. The HotpotQA dataset has one gold answer per question.
            predictions (list[str]): Predicted strings.

        Returns:
            float: Aggregated score over the current sample's items.
        """
        results = []
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(self, gold: str, pred: str) -> float:
        """
        Find the longest common subsequence between the gold and the prediction by fuzzy matching.
        """
        normalized_prediction = self._normalize_answer(pred)
        normalized_gold = self._normalize_answer(gold)

        if normalized_gold in ["yes", "no"]:
            return 1.0 if normalized_gold in normalized_prediction else 0.0
        
        if not normalized_gold:
            return 1.0 if not normalized_prediction else 0.0

        if not normalized_prediction:
            return 0.0

        prediction_tokens = normalized_prediction.split()
        gold_tokens = normalized_gold.split()

        # we want to find the longest common subsequence between the gold and the prediction by fuzzy matching
        m = len(gold_tokens)
        n = len(prediction_tokens)

        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                gold_token = gold_tokens[i - 1]
                pred_token = prediction_tokens[j - 1]

                similarity = self._edit_similarity(gold_token, pred_token)

                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + similarity)

        lcs_score = dp[m][n]
        import pdb; pdb.set_trace()
        return lcs_score / m


entailment_fuzzy_hotpotqa_v2 = SampleLevelMetric(
    metric_name="entailment_fuzzy_v2",
    sample_level_fn=Entailment_fuzzy_HotpotQA_v2().compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


class Entailment_exact_HotpotQA:
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
    ):
        self.aggregation_function = aggregation_function

    @staticmethod
    def _normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        results = []
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(self, gold: str, pred: str) -> float:
        normalized_prediction = self._normalize_answer(pred)
        normalized_gold = self._normalize_answer(gold)

        if normalized_gold in ["yes", "no"]:
            return 1.0 if normalized_gold in normalized_prediction else 0.0
        
        if not normalized_gold:
            return 1.0 if not normalized_prediction else 0.0

        if not normalized_prediction:
            return 0.0

        # find the longest common substring between the gold and the prediction
        m = len(normalized_gold)
        n = len(normalized_prediction)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if normalized_gold[i - 1] == normalized_prediction[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])
                else:
                    dp[i][j] = 0

        score = max_length / m
        return score
        # if score >= 0.5:
        #     return 1.0
        # else:
        #     return 0.0


entailment_exact_hotpotqa = SampleLevelMetric(
    metric_name="entailment_exact",
    sample_level_fn=Entailment_exact_HotpotQA().compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
