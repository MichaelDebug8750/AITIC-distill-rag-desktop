# -*- coding: utf-8 -*-
"""
test_pipeline.py — 关键逻辑单元测试（pytest）

覆盖不依赖 Ollama/模型的纯逻辑：
  · 计算器 safe_eval：正确求值 + 注入防御（重点）
  · 规则路由 route/extract_expr：calc vs retrieve 判定
  · 拒答判定 is_abstain：[NO REFERENCE FOUND] 及变体
  · 音频 _mmss / 模型路径解析
  · 语义分块 semantic_chunks / 分句

运行：pip install pytest  然后  pytest -v
（需 Ollama 的检索/生成不在此测，保持快、无外部依赖、可进 CI。）
"""
import os
import sys
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_runtime as ar
import asr

# main.py 顶部若无 ollama/chromadb/fitz 会 sys.exit / ImportError —— 兜住以便本机可测、CI 可跳
try:
    import main
    HAS_MAIN = True
except BaseException:
    main = None
    HAS_MAIN = False

needs_main = pytest.mark.skipif(not HAS_MAIN, reason="需 ollama/chromadb/fitz 环境（本机运行）")


# ============================== 计算器：正确性 ==============================
class TestCalculator:
    @pytest.mark.parametrize("expr,expected", [
        ("3*(4+5)", 27),
        ("2**10", 1024),
        ("100-58", 42),
        ("10/4", 2.5),
        ("17%5", 2),
        ("abs(-5)", 5),
    ])
    def test_arithmetic(self, expr, expected):
        assert ar.safe_eval(expr) == expected

    def test_float_and_funcs(self):
        assert ar.safe_eval("sqrt(144)") == pytest.approx(12.0)
        assert ar.safe_eval("log10(1000)") == pytest.approx(3.0)
        assert ar.safe_eval("100/7") == pytest.approx(14.2857, abs=1e-4)
        assert ar.safe_eval("pi") == pytest.approx(math.pi)
        assert ar.safe_eval("round(3.14159,2)") == pytest.approx(3.14)

    # ---------- 安全性：注入必须全部拒绝 ----------
    @pytest.mark.parametrize("payload", [
        "__import__('os').system('echo hi')",  # 导入执行
        "open('x')",                            # 文件操作
        "eval('2')",                            # 嵌套 eval
        "1 + abs.__class__",                    # 属性访问
        "[].append(1)",                         # 列表方法/下标
        "os.system('x')",                       # 未知名
        "x + 1",                                # 未定义标识符
    ])
    def test_rejects_injection(self, payload):
        with pytest.raises(Exception):
            ar.safe_eval(payload)


# ============================== 规则路由 ==============================
class TestRouting:
    @pytest.mark.parametrize("q", [
        "3*(4+5)",
        "12.5 * 8",
        "帮我算 3*(4+5) 等于多少",
        "sqrt(144)",
        "2**10 是多少",
        "100 / 7",
    ])
    def test_routes_to_calc(self, q):
        assert ar.route(q)[0] == "calc"

    @pytest.mark.parametrize("q", [
        "What are the parts of a bacteriophage?",
        "COVID-19 是什么",           # 含数字+连字符，不应误判
        "1918年发生了什么",           # 含数字无运算，不应误判
        "细胞膜的功能是什么",
        "What is a process in an operating system?",
    ])
    def test_routes_to_retrieve(self, q):
        assert ar.route(q)[0] == "retrieve"

    def test_extract_expr_returns_none_for_prose(self):
        assert ar.extract_expr("just a normal question") is None

    def test_extract_expr_normalizes_symbols(self):
        # × ÷ ^ 应被归一化后可求值
        assert ar.safe_eval(ar.extract_expr("6×7")) == 42
        assert ar.safe_eval(ar.extract_expr("2^8")) == 256


# ============================== 音频工具 ==============================
class TestAsr:
    @pytest.mark.parametrize("sec,mmss", [
        (0, "00:00"), (34, "00:34"), (155, "02:35"), (390, "06:30"), (3661, "61:01"),
    ])
    def test_mmss(self, sec, mmss):
        assert asr._mmss(sec) == mmss

    def test_resolve_model_found(self, tmp_path):
        d = tmp_path / "faster-whisper-small"
        d.mkdir()
        assert asr._resolve_model(str(d)) == str(d)

    def test_resolve_model_missing_raises(self, monkeypatch):
        monkeypatch.setattr(asr, "MODEL_CANDIDATES", [])
        with pytest.raises(FileNotFoundError):
            asr._resolve_model("/no/such/dir")


# ============================== main 纯函数（需环境） ==============================
@needs_main
class TestMainPure:
    @pytest.mark.parametrize("ans", [
        "[NO REFERENCE FOUND]",
        "no reference found",
        "The answer is [NO RELEVANT REFERENCE FOUND].",
        "NO REFERENCES FOUND",
        "",
        None,
    ])
    def test_is_abstain_true(self, ans):
        assert main.is_abstain(ans) is True

    @pytest.mark.parametrize("ans", [
        "A bacteriophage has a capsid and tail fibers [p.2].",
        "The process is an instance of a running program.",
    ])
    def test_is_abstain_false(self, ans):
        assert main.is_abstain(ans) is False

    def test_split_sentences(self):
        out = main.split_sentences("First sentence. Second one! Third?")
        assert len(out) == 3

    def test_semantic_chunks_basic(self):
        text = "p1: " + "This is a test sentence. " * 60
        chunks = main.semantic_chunks(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) and c for c in chunks)

    def test_semantic_chunks_empty(self):
        assert main.semantic_chunks("") == []

    def test_format_prompt_supplies_real_tag_example(self):
        metas = [{"page": 7, "type": "text", "source": "book.pdf"}]
        prompt = main._format_prompt("[p.7]\nsource block", "question?", [0], metas)
        assert "[p.7]" in prompt
        assert "question?" in prompt

    def test_verify_citations_requires_explicit_citation(self):
        metas = [{"page": 7, "type": "text", "source": "book.pdf"}]
        good = main.verify_citations("Supported answer [p.7].", [0], metas)
        missing = main.verify_citations("Supported answer without a citation.", [0], metas)
        abstain = main.verify_citations("[NO REFERENCE FOUND]", [0], metas)
        assert good["ok"] is True
        assert missing["ok"] is False and missing["missing"] is True
        assert abstain["ok"] is True and abstain["missing"] is False

    def test_subject_of_ignores_generic_data_dir(self):
        assert main._subject_of(r"E:\Ollama_test\data\book.pdf") is None
        assert main._subject_of(r"E:\Ollama_test\books\Medicine\book.pdf") == "Medicine"

    def test_resolve_gate_is_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(main, "read_manifest", lambda: {"subject": "medicine"})
        assert main.resolve_gate() == main.ESCALATE_GATE_BY_SUBJECT["Medicine"]

    def test_build_image_rasterizes_svg_before_vl(self, tmp_path, monkeypatch):
        """Ollama 不接收 SVG；build_image 必须把它转成 PNG 后再传给 VL。"""
        import base64

        svg_path = tmp_path / "flow.svg"
        svg_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="120">'
            '<rect width="320" height="120" fill="white"/>'
            '<text x="20" y="70" font-size="32">INPUT TO VECTOR DB</text>'
            '</svg>',
            encoding="utf-8",
        )

        class FakeCollection:
            def __init__(self):
                self.upsert_args = None

            def upsert(self, **kwargs):
                self.upsert_args = kwargs

        class FakeClient:
            def get_or_create_collection(self, _name):
                return collection

        collection = FakeCollection()
        seen = {}

        def fake_chat_vl(_model, _prompt, image_b64):
            image_bytes = base64.b64decode(image_b64)
            seen["image_bytes"] = image_bytes
            return "The diagram says INPUT TO VECTOR DB."

        monkeypatch.setattr(main.chromadb, "PersistentClient", lambda path: FakeClient())
        monkeypatch.setattr(main, "_chat_vl", fake_chat_vl)
        monkeypatch.setattr(main, "embed", lambda texts: [[0.0] for _ in texts])
        monkeypatch.setattr(main, "write_manifest", lambda **kwargs: None)
        monkeypatch.setattr(main, "_stamp_collection", lambda _col: None)

        main.build_image(str(svg_path))

        assert seen["image_bytes"].startswith(b"\x89PNG\r\n\x1a\n")
        assert collection.upsert_args["metadatas"][0]["type"] == "image"
        assert collection.upsert_args["metadatas"][0]["loc"] == "image:flow.svg"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
