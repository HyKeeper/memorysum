#!/usr/bin/env python3
"""从 scope 消息中提取事实，生成覆盖 20 个能力叶子的 QA 对。

用法：
    .venv/bin/python bench/generate_questions.py --target 200
    .venv/bin/python bench/generate_questions.py --scopes data/scopes --out bench/benchmark_v1.jsonl

生成的每道题都有确定性答案（answer_substrings），可在检索结果中做子串匹配自动评分。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

# ─── 事实提取器 ───────────────────────────────────────────────────────


def _msg_facts(text: str, role: str) -> list[dict]:
    """从单条消息中提取可验证的事实片段。"""
    facts: list[dict] = []
    # 名字
    for m in re.finditer(
        r"(?:my name is|I am|I'm|call me|名字是)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})",
        text,
    ):
        facts.append({"type": "name", "value": m.group(1), "span": m.group(0)})

    # 日期
    for m in re.finditer(
        r"(\d{4}-\d{2}-\d{2}|"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|"
        r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
        text,
    ):
        facts.append({"type": "date", "value": m.group(1), "span": m.group(0)})

    # 金额
    for m in re.finditer(r"\$[\d,]+(?:\.\d{2})?", text):
        facts.append({"type": "money", "value": m.group(0), "span": m.group(0)})

    # 百分比
    for m in re.finditer(r"\d+(?:\.\d+)?\s*%", text):
        facts.append({"type": "percentage", "value": m.group(0), "span": m.group(0)})

    # 偏好
    for m in re.finditer(
        r"(?:I prefer|I like|I love|I enjoy|my favorite|我(喜欢|偏好|最爱))\s+(.{5,60}?)(?:[.!?,]|$)",
        text,
    ):
        facts.append({"type": "preference", "value": m.group(0)[:80], "span": m.group(0)})

    # 身份/角色
    for m in re.finditer(
        r"(?:I am a|I work as|I'm a|my (?:job|role|title) is|I work at|I work for)\s+(.{3,50}?)(?:[.!?,]|$)",
        text,
    ):
        facts.append({"type": "identity", "value": m.group(0)[:80], "span": m.group(0)})

    # 健康
    for m in re.finditer(
        r"(?:I have|I was diagnosed with|I suffer from|I take|my (?:doctor|medication))\s+(.{5,60}?)(?:[.!?,]|$)",
        text,
    ):
        facts.append({"type": "health", "value": m.group(0)[:80], "span": m.group(0)})

    # 温度/数量/测量
    for m in re.finditer(r"\d+(?:\.\d+)?\s*(?:degrees?|km|miles?|kg|lbs?|hours?|minutes?|days?|weeks?|months?|years?)\b", text):
        facts.append({"type": "measurement", "value": m.group(0), "span": m.group(0)})

    # 规则/指令
    for m in re.finditer(
        r"(?:must|should|always|never|rule|policy|require[sd]?|make sure)\s+(.{10,80}?)(?:[.!?,]|$)",
        text,
        re.IGNORECASE,
    ):
        facts.append({"type": "rule", "value": m.group(0)[:80], "span": m.group(0)})

    # 因果关系
    for m in re.finditer(
        r"(?:because|since|therefore|as a result|caused by|led to|resulted in)\s+(.{10,80}?)(?:[.!?,]|$)",
        text,
        re.IGNORECASE,
    ):
        facts.append({"type": "causal", "value": m.group(0)[:80], "span": m.group(0)})

    return facts


def _find_temporal_pairs(messages: list[dict]) -> list[dict]:
    """找包含相对时间的消息，结合时间戳生成 C1 题。"""
    pairs = []
    relative_patterns = [
        (r"yesterday", -1),
        (r"last week", -7),
        (r"last month", -30),
        (r"last year", -365),
        (r"(?:two|three|four|five)\s+(?:days?|weeks?|months?|years?)\s+ago", None),
        (r"(?:a few|several|couple of)\s+(?:days?|weeks?|months?|years?)\s+ago", None),
        (r"(?:next|coming)\s+(?:week|month|year)", 1),
    ]
    for msg in messages:
        text = msg.get("raw_content", "") or ""
        ts_ms = msg.get("source_timestamp_ms")
        ts_str = msg.get("timestamp_provided", "")
        for pattern, _offset in relative_patterns:
            if re.search(pattern, text, re.IGNORECASE) and ts_ms:
                pairs.append({"msg": msg, "relative": pattern, "ts_ms": ts_ms, "ts_str": ts_str})
                break
    return pairs


def _find_speaker_pairs(messages: list[dict]) -> list[dict]:
    """找说话人归因对（A2 题）。"""
    pairs = []
    for msg in messages:
        text = msg.get("raw_content", "") or ""
        role = msg.get("role", "user")
        if role == "assistant" and len(text) > 30:
            # 找 assistant 消息中的第一句有意义的陈述
            first_sentence = re.split(r"[.!?\n]", text)[0][:80]
            if len(first_sentence) > 15:
                pairs.append({"role": role, "text": first_sentence, "full": text[:200]})
    return pairs


def _find_update_pairs(messages: list[dict]) -> list[dict]:
    """找覆盖/更新对（D1 题）：同一实体出现新旧两个值。"""
    pairs = []
    # 简单策略：找后一条消息修正前一条消息的情况
    # 通过 "actually", "no wait", "I mean", "correction", "update" 等关键词
    update_markers = [
        r"(?:actually|no[,\s]+wait|I mean|correction|update|instead|rather than|changed to|moved to|now)\s+(.{10,80})",
    ]
    for i, msg in enumerate(messages):
        text = msg.get("raw_content", "") or ""
        for pattern in update_markers:
            m = re.search(pattern, text, re.IGNORECASE)
            if m and i > 0:
                prev_text = messages[i - 1].get("raw_content", "") or ""
                if len(prev_text) > 20:
                    pairs.append(
                        {
                            "old_msg_idx": i - 1,
                            "new_msg_idx": i,
                            "new_value": m.group(0)[:80],
                            "old_excerpt": prev_text[:100],
                        }
                    )
    return pairs


def _find_cross_session_facts(messages: list[dict]) -> list[dict]:
    """找跨 session 的多跳线索（B1 题）。"""
    pairs = []
    # 按 session 分组
    sessions: dict[str, list[dict]] = {}
    for msg in messages:
        content = msg.get("raw_content", "") or ""
        # 尝试从 META 头提取 session
        sm = re.search(r"session[_\s]?(\d+)", content[:200], re.IGNORECASE)
        sess_key = sm.group(1) if sm else "0"
        sessions.setdefault(sess_key, []).append(msg)

    if len(sessions) < 2:
        return pairs

    # 找跨 session 出现的共同实体（人名、项目名）
    entity_pattern = r"\b([A-Z][a-z]{2,15})\b"
    session_entities: dict[str, set[str]] = {}
    for sess_key, msgs in sessions.items():
        entities = set()
        for msg in msgs[:20]:  # 限制扫描量
            text = msg.get("raw_content", "") or ""
            for em in re.finditer(entity_pattern, text):
                w = em.group(1)
                if w not in {"The", "This", "That", "There", "They", "Their", "What", "When", "Where", "How", "But", "And", "For", "Not", "You", "All", "Can", "Had", "Her", "His", "Its", "Let", "May", "New", "Now", "Our", "Out", "Own", "Say", "She", "Too", "Use", "Way", "Will", "With", "Yes", "After", "Also", "Back", "Been", "Both", "Call", "Come", "Could", "Each", "Even", "Find", "From", "Give", "Good", "Have", "Here", "High", "Just", "Keep", "Know", "Last", "Like", "Long", "Look", "Made", "Make", "Many", "More", "Most", "Much", "Must", "Name", "Need", "Only", "Other", "Over", "Part", "Same", "Said", "Show", "Some", "Such", "Take", "Than", "Them", "Then", "These", "They", "Time", "Very", "Want", "Well", "Went", "Were", "What", "When", "Will", "Work", "Your"}:
                    entities.add(w)
        session_entities[sess_key] = entities

    # 找在 >= 2 个 session 中出现的实体
    all_keys = list(session_entities.keys())
    for i in range(len(all_keys)):
        for j in range(i + 1, len(all_keys)):
            shared = session_entities[all_keys[i]] & session_entities[all_keys[j]]
            for entity in list(shared)[:3]:
                pairs.append(
                    {
                        "entity": entity,
                        "sessions": [all_keys[i], all_keys[j]],
                    }
                )
    return pairs[:10]  # 限制数量


# ─── 题目生成 ─────────────────────────────────────────────────────────


class QuestionGenerator:
    """从 scope 数据生成 QA 对。"""

    def __init__(self, scope_dir: Path):
        self.scope_dir = scope_dir
        self.questions: list[dict] = []
        self._q_counter = 0

    def _next_id(self, leaf: str) -> str:
        self._q_counter += 1
        return f"{leaf}-{self._q_counter:03d}"

    def _add(self, leaf: str, scope_hash: str, question: str, answers: list[str],
             score_type: str = "substring", evidence: list[int] | None = None, max_per_leaf: int = 15, **extra):
        # 叶子级别限制：避免某个叶子过度集中
        leaf_count = sum(1 for q in self.questions if q["leaf"] == leaf)
        if leaf_count >= max_per_leaf:
            return
        q = {
            "id": self._next_id(leaf),
            "leaf": leaf,
            "scope_hash": scope_hash,
            "question": question,
            "answer_substrings": answers,
            "score_type": score_type,
            "evidence_msg_ids": evidence or [],
            **extra,
        }
        self.questions.append(q)

    def generate_from_scope(self, scope_hash: str, messages: list[dict], user_id: str = "") -> int:
        """从一个 scope 的消息列表生成题目。返回生成的题目数。"""
        before = len(self.questions)
        valid_msgs = [
            m for m in messages
            if (m.get("raw_content") or "").strip()
            and len(m.get("raw_content") or "") > 20
        ]
        if len(valid_msgs) < 5:
            return 0

        # 用 user_id 作为 scope 标识（如果有），否则用 hash
        scope_id = user_id or scope_hash

        # A1: 单点事实
        for msg in valid_msgs[:60]:
            for fact in _msg_facts(msg["raw_content"], msg.get("role", "user")):
                if fact["type"] == "name":
                    self._add("A1", scope_id,
                              f"What is the speaker's name mentioned in this conversation?",
                              [fact["value"]], evidence=[msg.get("ordinal", 0)])
                elif fact["type"] == "money":
                    self._add("A1", scope_id,
                              f"What monetary amount is mentioned?",
                              [fact["value"]], evidence=[msg.get("ordinal", 0)])
                elif fact["type"] == "identity":
                    self._add("A1", scope_id,
                              f"What is the speaker's job or role?",
                              [fact["value"][:60]], evidence=[msg.get("ordinal", 0)])
                elif fact["type"] == "health":
                    self._add("A1", scope_id,
                              f"What health-related information is mentioned?",
                              [fact["value"][:60]], evidence=[msg.get("ordinal", 0)])
                elif fact["type"] == "measurement":
                    self._add("A1", scope_id,
                              f"What measurement or quantity is mentioned?",
                              [fact["value"]], evidence=[msg.get("ordinal", 0)])
                elif fact["type"] == "date":
                    self._add("A1", scope_id,
                              f"What specific date is mentioned in the conversation?",
                              [fact["value"]], evidence=[msg.get("ordinal", 0)])

        # A2: 说话人归因
        for pair in _find_speaker_pairs(valid_msgs)[:15]:
            self._add("A2", scope_id,
                      f"What does the assistant say about the topic being discussed?",
                      [pair["text"]], evidence=[])

        # E1: 偏好（扩展模式）
        preference_patterns = [
            r"(?:I prefer|I like|I love|I enjoy|my favorite|my fav|I'm into|I'm a fan of|I usually|I tend to|I always|I never)\s+(.{5,80}?)(?:[.!?,]|$)",
            r"(?:my favorite|最喜欢的|我偏好)\s+(.{5,60}?)(?:[.!?,]|$)",
            r"(?:rather|would rather|I'd rather|prefer to)\s+(.{5,60}?)(?:[.!?,]|$)",
        ]
        for msg in valid_msgs[:60]:
            text = msg.get("raw_content", "") or ""
            for pattern in preference_patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    pref = m.group(0)[:80]
                    if len(pref) > 10:
                        self._add("E1", scope_id,
                                  f"What preference or favorite thing does the speaker mention?",
                                  [pref], evidence=[msg.get("ordinal", 0)])
                        break  # 一条消息只取一个

        # E2: 刻板/反刻板偏好（从偏好推断是否刻板）
        for msg in valid_msgs[:40]:
            text = msg.get("raw_content", "") or ""
            # 找 "actually", "surprisingly", "unlike most" 等反刻板标记
            anti_pattern = re.search(
                r"(?:actually|surprisingly|unlike most|people think I'm|you might think|not what you'd expect)\s+(.{10,80}?)(?:[.!?,]|$)",
                text, re.IGNORECASE)
            if anti_pattern:
                self._add("E2", scope_id,
                          f"What surprising or non-stereotypical preference does the speaker have?",
                          [anti_pattern.group(0)[:60]], evidence=[msg.get("ordinal", 0)])

        # E3: 个人背景/健康
        health_patterns = [
            r"(?:I have|I was diagnosed with|I suffer from|I take|my (?:doctor|medication)|I'm (?:allergic|sensitive) to|I can't eat|I'm (?:vegetarian|vegan|lactose))\s+(.{5,60}?)(?:[.!?,]|$)",
            r"(?:my (?:age|birthday|birth date|born) (?:is|was))\s+(.{5,40}?)(?:[.!?,]|$)",
            r"(?:I live in|I'm from|I grew up in|my hometown)\s+(.{5,40}?)(?:[.!?,]|$)",
        ]
        for msg in valid_msgs[:60]:
            text = msg.get("raw_content", "") or ""
            for pattern in health_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    self._add("E3", scope_id,
                              f"What personal background or health information is shared?",
                              [m.group(0)[:60]], evidence=[msg.get("ordinal", 0)])

        # C1: 时间题
        for pair in _find_temporal_pairs(valid_msgs)[:10]:
            ts_str = pair.get("ts_str", "")
            self._add("C1", scope_id,
                      f"When does the event with relative time reference occur? (answer should include the absolute date)",
                      [pair["relative"], ts_str] if ts_str else [pair["relative"]],
                      score_type="temporal",
                      evidence=[pair["msg"].get("ordinal", 0)])

        # C2: 事件顺序
        if len(valid_msgs) >= 3:
            # 取三条有日期的消息，问顺序
            dated = [
                (i, m) for i, m in enumerate(valid_msgs)
                if m.get("source_timestamp_ms") and re.search(
                    r"\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}",
                    m.get("raw_content", ""),
                )
            ]
            if len(dated) >= 2:
                sorted_dated = sorted(dated, key=lambda x: x[1]["source_timestamp_ms"])
                first_text = re.split(r"[.!?\n]", sorted_dated[0][1]["raw_content"])[0][:60]
                last_text = re.split(r"[.!?\n]", sorted_dated[-1][1]["raw_content"])[0][:60]
                if len(first_text) > 10 and len(last_text) > 10:
                    self._add("C2", scope_id,
                              f"In what order do these events occur?",
                              [first_text, last_text],
                              score_type="multi_evidence",
                              evidence=[sorted_dated[0][0], sorted_dated[-1][0]])

        # C3: 状态变化
        # 同一实体在不同消息中出现不同状态
        state_patterns = [
            (r"(?:now|currently|these days)\s+(.{10,60})", "current"),
            (r"(?:previously|before|used to|earlier)\s+(.{10,60})", "previous"),
        ]
        for msg in valid_msgs[:40]:
            text = msg["raw_content"]
            states = {}
            for pattern, label in state_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    states[label] = m.group(0)[:60]
            if len(states) == 2:
                self._add("C3", scope_id,
                          f"How does the speaker's situation change over time?",
                          [states.get("previous", ""), states.get("current", "")],
                          score_type="multi_evidence",
                          evidence=[msg.get("ordinal", 0)])

        # D1: 新值覆盖
        for pair in _find_update_pairs(valid_msgs)[:8]:
            self._add("D1", scope_id,
                      f"What is the updated or corrected value mentioned in the conversation?",
                      [pair["new_value"][:60]],
                      evidence=[pair["new_msg_idx"]])

        # B1: 跨 session 聚合
        # 从 LoCoMo 风格的多 session 数据中生成跨 session 问题
        session_pattern = r"Session\s+(\d+)\s*\(([^)]+)\)"
        sessions_found = []
        for msg in valid_msgs:
            text = msg.get("raw_content", "") or ""
            sm = re.search(session_pattern, text)
            if sm:
                sessions_found.append({"num": sm.group(1), "time": sm.group(2), "text": text[:300]})
        if len(sessions_found) >= 2:
            # 找在多个 session 中提到的实体
            first_text = sessions_found[0]["text"]
            last_text = sessions_found[-1]["text"]
            # 提取人名
            names = set()
            for sf_msg in valid_msgs:
                for nm in re.finditer(r"\b([A-Z][a-z]{2,12})\b", sf_msg.get("raw_content", "") or ""):
                    w = nm.group(1)
                    if w not in {"Session", "The", "This", "That", "There", "They", "What", "When", "Where", "How", "And", "But", "For", "Not", "You", "All", "Can", "Had", "Her", "His", "Its", "Let", "May", "New", "Now", "Our", "Out", "Own", "Say", "She", "Too", "Use", "Way", "Will", "With", "Yes", "Also", "Back", "Been", "Both", "Call", "Come", "Each", "Even", "Find", "From", "Give", "Good", "Have", "Here", "Just", "Keep", "Know", "Last", "Like", "Long", "Look", "Made", "Make", "Many", "More", "Most", "Much", "Must", "Name", "Need", "Only", "Other", "Over", "Part", "Same", "Said", "Show", "Some", "Such", "Take", "Than", "Them", "Then", "These", "They", "Time", "Very", "Want", "Well", "Went", "Were", "What", "When", "Will", "Work", "Your", "Image", "Hey", "Sure", "Thanks", "Thank", "Great", "Yeah", "Okay", "Well", "Right", "Actually", "Anyway", "Absolutely", "Congratulations", "Congrats", "Mel", "User", "Assistant"}:
                        names.add(w)
            for name in list(names)[:5]:
                self._add("B1", scope_id,
                          f"What do we learn about {name} across different sessions in this conversation?",
                          [name],
                          score_type="multi_evidence")
            # 直接问跨 session 的事实
            self._add("B1", scope_id,
                      f"What events or facts are mentioned across multiple sessions?",
                      [sessions_found[0].get("time", ""), sessions_found[-1].get("time", "")],
                      score_type="multi_evidence")
        else:
            for pair in _find_cross_session_facts(valid_msgs)[:8]:
                self._add("B1", scope_id,
                          f"What information about '{pair['entity']}' appears across multiple sessions?",
                          [pair["entity"]],
                          score_type="multi_evidence")

        # B2: 因果链
        for msg in valid_msgs[:40]:
            for fact in _msg_facts(msg["raw_content"], msg.get("role", "user")):
                if fact["type"] == "causal":
                    self._add("B2", scope_id,
                              f"What cause-and-effect relationship is described?",
                              [fact["value"][:60]],
                              evidence=[msg.get("ordinal", 0)])

        # B4: 开放域/建议
        advice_patterns = [
            r"(?:recommend|suggest|advice|should try|might want to|consider)\s+(.{10,80})",
            r"(?:the best (?:way|approach|method) (?:is|would be))\s+(.{10,80})",
            r"(?:I would|I'd|you should|you could|try to|make sure to)\s+(.{10,80}?)(?:[.!?,]|$)",
            r"(?:don't forget|remember to|be sure to|make sure you)\s+(.{10,80}?)(?:[.!?,]|$)",
        ]
        for msg in valid_msgs[:40]:
            text = msg.get("raw_content", "") or ""
            for pattern in advice_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m and len(m.group(0)) > 15:
                    self._add("B4", scope_id,
                              f"What recommendation, suggestion, or advice is given?",
                              [m.group(0)[:60]],
                              evidence=[msg.get("ordinal", 0)])
                    break

        # D2: 矛盾检测（找 "actually"/"wait"/"correction" 等修正标记）
        correction_patterns = [
            r"(?:actually|no[,.]?\s*wait|I mean|correction|that's not right|I was wrong|turns out)\s+(.{10,80}?)(?:[.!?,]|$)",
            r"(?:I changed my mind|on second thought|never mind|forget what I said)\s+(.{5,60}?)(?:[.!?,]|$)",
        ]
        for msg in valid_msgs[:40]:
            text = msg.get("raw_content", "") or ""
            for pattern in correction_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    self._add("D2", scope_id,
                              f"What correction or contradiction is stated?",
                              [m.group(0)[:60]],
                              evidence=[msg.get("ordinal", 0)])

        # G2: 规则系统
        for msg in valid_msgs[:40]:
            for fact in _msg_facts(msg["raw_content"], msg.get("role", "user")):
                if fact["type"] == "rule":
                    self._add("G2", scope_id,
                              f"What rule or requirement is stated?",
                              [fact["value"][:60]],
                              evidence=[msg.get("ordinal", 0)])

        # G4: 经验数据发现（扩展模式）
        data_patterns = [
            r"(\d+(?:\.\d+)?\s*%)",
            r"(\d+(?:\.\d+)?\s*(?:degrees?|km|miles?|kg|lbs?|hours?|minutes?|days?|weeks?|months?|years?))",
            r"((?:increased|decreased|grew|dropped|rose|fell|improved|declined)\s+(?:by\s+)?\d+(?:\.\d+)?\s*%)",
            r"(\$[\d,]+(?:\.\d{2})?)",
        ]
        for msg in valid_msgs[:50]:
            text = msg.get("raw_content", "") or ""
            for pattern in data_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    self._add("G4", scope_id,
                              f"What quantitative data point or statistic is mentioned?",
                              [m.group(1) if m.lastindex else m.group(0)],
                              evidence=[msg.get("ordinal", 0)])

        # I1: 在线学习（新事实覆盖旧事实）
        # 找后面的消息纠正/补充前面消息的情况
        for pair in _find_update_pairs(valid_msgs)[:5]:
            self._add("I1", scope_id,
                      f"What is the most recent value or status mentioned?",
                      [pair["new_value"][:60]],
                      evidence=[pair["new_msg_idx"]])

        # I2: 内部迁移（跨 session/跨话题的信息复用）
        # 找在不同 session 中被引用或重复的信息
        if len(sessions_found) >= 2:
            # 找第一个 session 中提到的事实，看是否在后面 session 中被引用
            self._add("I2", scope_id,
                      f"What information from an earlier session is referenced or continued in a later session?",
                      [sessions_found[0].get("time", "")],
                      score_type="multi_evidence")
        else:
            for pair in _find_cross_session_facts(valid_msgs)[:5]:
                self._add("I2", scope_id,
                          f"What information is shared or transferred across different contexts?",
                          [pair["entity"]],
                          score_type="multi_evidence")

        # G1: 领域知识
        for msg in valid_msgs[:30]:
            text = msg["raw_content"]
            # 找技术术语 + 定义
            m = re.search(r"(\w+)\s+(?:is|are|means?|refers to)\s+(.{15,80}?)(?:[.!?,]|$)", text)
            if m and len(m.group(2)) > 15:
                self._add("G1", scope_id,
                          f"What is the definition or explanation of '{m.group(1)}'?",
                          [m.group(0)[:60]],
                          evidence=[msg.get("ordinal", 0)])

        # H2: 隐私最小披露（扩展模式：任何个人信息都算）
        sensitive_patterns = [
            r"(?:password|secret|token|api[_\s]?key|credential|ssn|social security)\s*(?:is|:|=)\s*(.{5,40})",
            r"(?:credit card|bank account|routing number)\s*(?:is|:|=)\s*(.{5,40})",
            r"(?:my email|my phone|my address)\s*(?:is|:|=)\s*(.{5,40})",
            r"(?:my (?:full )?name)\s*(?:is|:|=)\s*(.{5,40})",
            r"(?:born|birthday|birth date)\s*(?:on|is|was|:|=)\s*(.{5,40})",
            r"(?:my (?:address|phone|email|contact))\s*(?:is|:|=)\s*(.{5,40})",
            r"(?:I live at|my address is)\s+(.{5,60}?)(?:[.!?,]|$)",
            r"(?:my number|call me at|reach me at)\s+(.{5,40}?)(?:[.!?,]|$)",
            # 任何包含个人身份信息的句子
            r"(?:my (?:wife|husband|partner|son|daughter|mother|father|brother|sister))\s+(.{5,60}?)(?:[.!?,]|$)",
        ]
        for msg in valid_msgs[:40]:
            text = msg.get("raw_content", "") or ""
            for pattern in sensitive_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    self._add("H2", scope_id,
                              f"What sensitive or personal information is mentioned that should be handled carefully?",
                              [m.group(0)[:60]],
                              evidence=[msg.get("ordinal", 0)])
                    break
        # 兜底：如果一条都没找到，从任意消息中提取个人信息类事实
        if not any(q["leaf"] == "H2" for q in self.questions):
            for msg in valid_msgs[:20]:
                text = msg.get("raw_content", "") or ""
                # 找名字 + 身份组合
                m = re.search(r"\b([A-Z][a-z]+)\s+(?:is|works at|lives in|was born)\s+(.{5,60}?)(?:[.!?,]|$)", text)
                if m:
                    self._add("H2", scope_id,
                              f"What personal information about someone is mentioned?",
                              [m.group(0)[:60]],
                              evidence=[msg.get("ordinal", 0)])
                    break

        # F1: 摘要/全局综合
        # 从多条消息中提取关键事实，要求覆盖率
        if len(valid_msgs) >= 8:
            key_facts = []
            for msg in valid_msgs[:15]:
                text = msg["raw_content"]
                # 提取每个消息的关键词（名词短语）
                words = re.findall(r"\b[A-Z][a-z]{3,15}\b", text)
                if words:
                    key_facts.append(words[0])
            if len(key_facts) >= 4:
                self._add("F1", scope_id,
                          f"Provide a comprehensive summary of the key topics discussed.",
                          key_facts[:8],
                          score_type="coverage")

        return len(self.questions) - before

    def load_scope_messages(self, scope_file: Path) -> list[dict]:
        """从 scope sqlite 文件读取消息。"""
        try:
            conn = sqlite3.connect(str(scope_file))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ordinal, role, raw_content, source_timestamp_ms, "
                "timestamp_provided, message_id FROM messages ORDER BY ordinal"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []


# ─── 主流程 ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="生成基准测试题库")
    parser.add_argument("--target", type=int, default=200, help="目标题数")
    parser.add_argument("--scopes", type=str, default="data/scopes", help="scope 数据目录")
    parser.add_argument("--out", type=str, default="bench/benchmark_v1.jsonl", help="输出文件")
    parser.add_argument("--min-msgs", type=int, default=10, help="最少消息数")
    args = parser.parse_args()

    scope_dir = Path(args.scopes)
    if not scope_dir.exists():
        print(f"错误：scope 目录不存在：{scope_dir}", file=sys.stderr)
        sys.exit(1)

    # 收集 scope 文件
    scope_files = sorted(
        f for f in scope_dir.iterdir()
        if f.suffix == ".sqlite3" and "-wal" not in f.name and "-shm" not in f.name
    )
    print(f"找到 {len(scope_files)} 个 scope 文件")

    gen = QuestionGenerator(scope_dir)

    # 优先用内容丰富的 scope
    scored_scopes = []
    for sf in scope_files:
        msgs = gen.load_scope_messages(sf)
        if len(msgs) >= args.min_msgs:
            text_len = sum(len(m.get("raw_content") or "") for m in msgs)
            has_dates = sum(
                1 for m in msgs
                if re.search(r"\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)", m.get("raw_content") or "")
            )
            has_numbers = sum(1 for m in msgs if re.search(r"\d+", m.get("raw_content") or ""))
            # 优先：消息多、有日期、有数字
            score = len(msgs) * 2 + has_dates * 3 + has_numbers
            scored_scopes.append((score, sf, msgs))

    scored_scopes.sort(key=lambda x: -x[0])
    print(f"有效 scope: {len(scored_scopes)}")

    # 逐步生成，直到达到目标题数
    used_scopes = 0
    for score, sf, msgs in scored_scopes:
        if len(gen.questions) >= args.target:
            break
        scope_hash = sf.stem[:16]
        # 读取 user_id
        try:
            conn = sqlite3.connect(str(sf))
            row = conn.execute('SELECT user_id FROM scope_manifest LIMIT 1').fetchone()
            conn.close()
            uid = row[0] if row else ""
        except Exception:
            uid = ""
        n = gen.generate_from_scope(scope_hash, msgs, user_id=uid)
        if n > 0:
            used_scopes += 1
            print(f"  [{used_scopes:>3d}] {scope_hash}...  {len(msgs):>4} msgs → +{n} questions (total: {len(gen.questions)})")

    # 如果还不够，用低分 scope 补充
    if len(gen.questions) < args.target:
        for score, sf, msgs in scored_scopes[used_scopes:]:
            if len(gen.questions) >= args.target:
                break
            scope_hash = sf.stem[:16]
            try:
                conn = sqlite3.connect(str(sf))
                row = conn.execute('SELECT user_id FROM scope_manifest LIMIT 1').fetchone()
                conn.close()
                uid = row[0] if row else ""
            except Exception:
                uid = ""
            n = gen.generate_from_scope(scope_hash, msgs, user_id=uid)
            if n > 0:
                used_scopes += 1

    # 叶子覆盖统计
    from collections import Counter
    leaf_counts = Counter(q["leaf"] for q in gen.questions)
    print(f"\n生成完成：{len(gen.questions)} 题，来自 {used_scopes} 个 scope")
    print("\n叶子覆盖：")
    all_leaves = ["A1","A2","B1","B2","B4","C1","C2","C3","D1","D2","E1","E2","E3","F1","G1","G2","G4","H2","I1","I2"]
    for leaf in all_leaves:
        cnt = leaf_counts.get(leaf, 0)
        bar = "█" * min(cnt, 30)
        status = "✅" if cnt >= 5 else ("⚠️" if cnt > 0 else "❌")
        print(f"  {status} {leaf:4s} {cnt:>3d}  {bar}")
    missing = [l for l in all_leaves if leaf_counts.get(l, 0) == 0]
    if missing:
        print(f"\n❌ 缺失叶子: {', '.join(missing)}")

    # 写入
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for q in gen.questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
