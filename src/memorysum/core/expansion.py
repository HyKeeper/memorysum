"""伪相关反馈（pseudo-relevance feedback）的扩展词提取。

为什么要它：平台的 `top_k=100`，而我们返回中位 10~16 条；有些题目的答案，用原始查询
的措辞**就是召不到**（同一个事实在库里换了说法）。经典的 RM3/Rocchio 思路是——拿初始
检索的高分文档当"伪相关集"，从里面学出扩展词，再检索一轮。

**边界**（用户的硬约束）：**全程不用任何生成式模型**。这里只有词频、文档频次、以及现有的
embedding 服务；没有任何一步会"生成"文本。

与 AGENTS.md 第 18 条的关系：那条测的是**机械切分**（去停用词、切前后半）然后合并，
结论是"规则变体破坏语义、合并又稀释原排序"。本模块的区别是扩展词**从检索结果里学**
（信息量大得多），而且调用方必须用"并集 + 原查询重排"来保证原排序不被稀释。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence

from memorysum.core.text import chinese_terms, english_terms

#: 这些词在任何语料里都高频，作为扩展词只会引入噪声。只保留功能词，不做语义筛选——
#: 语义筛选需要模型，而这里刻意不用。
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "was",
        "were",
        "are",
        "have",
        "has",
        "had",
        "from",
        "they",
        "them",
        "their",
        "there",
        "here",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "how",
        "why",
        "did",
        "does",
        "done",
        "been",
        "being",
        "but",
        "not",
        "you",
        "your",
        "yours",
        "his",
        "her",
        "hers",
        "its",
        "our",
        "ours",
        "she",
        "him",
        "one",
        "two",
        "all",
        "any",
        "some",
        "such",
        "than",
        "then",
        "into",
        "onto",
        "about",
        "would",
        "could",
        "should",
        "will",
        "shall",
        "can",
        "may",
        "might",
        "must",
        "also",
        "more",
        "most",
        "much",
        "very",
        "just",
        "only",
        "other",
        "another",
        "same",
        "said",
        "says",
        "say",
        "get",
        "got",
        "make",
        "made",
        "like",
        "well",
        "back",
        "even",
        "still",
        "over",
        "under",
        "after",
        "before",
        "while",
        "because",
        "since",
        "though",
        "yes",
        "yeah",
        "okay",
        "ok",
        "sure",
        "thing",
        "things",
        "way",
        "ways",
        "time",
        "times",
        "的",
        "了",
        "是",
        "在",
        "我",
        "你",
        "他",
        "她",
        "它",
        "们",
        "和",
        "与",
        "就",
        "都",
        "而",
        "及",
        "或",
        "一个",
        "没有",
        "自己",
        "这个",
        "那个",
        "什么",
        "怎么",
        "为什么",
        "可以",
        "因为",
        "所以",
        "但是",
        "如果",
        "已经",
        "还是",
        "这样",
        "那样",
        "不是",
    }
)


def tokenize(text: str, *, limit: int = 400) -> list[str]:
    """英文 + 中文分词（复用检索通道用的同一套），去停用词与短词。"""
    terms = english_terms(text, limit=limit) + chinese_terms(text, limit=limit)
    output: list[str] = []
    for term in terms:
        term = term.strip().lower()
        if not term or term in _STOPWORDS:
            continue
        # 中文按 2 字起算（bigram 本来就是 2 字），英文要求 3 字符以上以免噪声
        if term.isascii() and len(term) < 3:
            continue
        if not term.isascii() and len(term) < 2:
            continue
        output.append(term)
    return output


def expansion_weights(
    texts: Iterable[str],
    query: str,
    df_lookup: Callable[[str], int],
    *,
    limit: int,
    max_df_ratio: float = 0.35,
    total_windows: int = 0,
) -> list[tuple[str, float]]:
    """按 ``TF / (1 + DF)`` 给候选扩展词打分，返回前 ``limit`` 个（含分数）。

    · `TF` 统计在**伪相关文档集合**里出现的文档数（document frequency within the set，
      比原始词频更抗长文），
    · `DF` 是词在**整个窗口库**里的窗口数，由调用方用 FTS 索引查（不扫库），
    · `max_df_ratio` 丢掉"满库都是"的词：那种词区分度为零，加进查询只会把检索带平。
    """
    if not texts:
        return []
    query_terms = set(tokenize(query))
    in_set = Counter()
    for text in texts:
        for term in set(tokenize(text)):
            in_set[term] += 1
    scored: list[tuple[float, str]] = []
    for term, tf in in_set.items():
        if term in query_terms:
            continue
        df = df_lookup(term)
        if df <= 0:
            continue
        if total_windows and df / total_windows > max_df_ratio:
            continue
        scored.append((tf / (1.0 + df), term))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(term, round(weight, 8)) for weight, term in scored[:limit]]


def blend_query_vector(
    base: Sequence[float],
    term_vectors: Sequence[Sequence[float]],
    alpha: float,
) -> list[float]:
    """把扩展词的向量插进查询向量：``v' = normalize(v_q + alpha * mean(v_w))``。

    为什么不是把词拼进查询文本：embedding 模型对拼接出来的"权重"毫无感知——`α` 只写在
    提示里不会改变向量。插值才让 `α` 真正起作用，而且**取平均**（不是求和）使 `α` 的语义
    与扩展词个数 `m` 解耦，调参时两个维度互不干扰。
    """
    if not term_vectors or alpha <= 0:
        return list(base)
    dimension = len(base)
    mean = [0.0] * dimension
    used = 0
    for vector in term_vectors:
        if len(vector) != dimension:
            continue
        for index, value in enumerate(vector):
            mean[index] += float(value)
        used += 1
    if not used:
        return list(base)
    blended = [float(base[index]) + alpha * (mean[index] / used) for index in range(dimension)]
    norm = sum(value * value for value in blended) ** 0.5
    if norm <= 0:
        return list(base)
    return [value / norm for value in blended]
