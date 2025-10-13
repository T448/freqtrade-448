#!/usr/bin/env python3
"""データリーク自動検出スクリプト（CI/CD統合用）

このスクリプトは、Python ASTを使用してソースコード内の
データリークパターン（.shift(-n)）を検出します。

使用方法:
    python scripts/detect_data_leak.py

終了コード:
    0: データリークなし（成功）
    1: データリーク検出（失敗）

CI/CD統合:
    このスクリプトをGitHub ActionsやCI/CDパイプラインで実行することで、
    自動的にデータリークを検出できます。
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Optional


class DataLeakDetector(ast.NodeVisitor):
    """ASTベースのデータリーク検出器

    .shift(-n)パターンを検出し、関数名と行番号を記録する
    """

    def __init__(self):
        """初期化"""
        self.violations: List[Dict[str, any]] = []
        self.current_function: Optional[str] = None
        self.function_stack: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """関数定義の訪問

        現在の関数名をスタックに記録する
        """
        self.function_stack.append(node.name)
        self.current_function = node.name
        self.generic_visit(node)
        self.function_stack.pop()
        self.current_function = self.function_stack[-1] if self.function_stack else None

    def visit_Call(self, node: ast.Call):
        """関数呼び出しの訪問

        .shift(-n)パターンを検出する
        """
        # .shift()メソッド呼び出しをチェック
        if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
            # 引数をチェック
            if node.args:
                arg = node.args[0]
                # 負の数値リテラル: -n
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    if isinstance(arg.operand, (ast.Constant, ast.Num)):
                        self._record_violation(node)
                # 負の定数: 変数に-nが代入されている場合は検出困難
                # （静的解析の限界）

        self.generic_visit(node)

    def _record_violation(self, node: ast.AST):
        """違反を記録

        Args:
            node: 違反が検出されたASTノード
        """
        self.violations.append(
            {
                "line": node.lineno,
                "col": node.col_offset,
                "function": self.current_function or "module_level",
            }
        )


def detect_shift_negative(file_path: Path) -> List[Dict[str, any]]:
    """Pythonファイル内の.shift(-n)を検出

    Args:
        file_path: 検査対象のPythonファイルパス

    Returns:
        違反のリスト。各違反は以下の情報を含む辞書:
        - line: 行番号
        - col: 列番号
        - function: 関数名

    Raises:
        SyntaxError: ファイルの構文エラー
        FileNotFoundError: ファイルが存在しない
    """
    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"⚠️  Syntax error in {file_path}: {e}")
        return []

    detector = DataLeakDetector()
    detector.visit(tree)

    return detector.violations


def is_allowed_function(function_name: str, file_path: Path) -> bool:
    """特定の関数での.shift(-n)使用が許可されているかチェック

    calculate_returns()など、未来データを正当に使用する関数では
    .shift(-n)の使用が許可される

    Args:
        function_name: 関数名
        file_path: ファイルパス

    Returns:
        許可されている場合True
    """
    # calculate_returns()では未来データ使用が正当
    if function_name == "calculate_returns":
        return True

    # _calculate_chase_returns(), _calculate_one_candle_returns()も許可
    if function_name in ["_calculate_chase_returns", "_calculate_one_candle_returns"]:
        return True

    return False


def scan_files(file_patterns: List[str]) -> Dict[str, List[Dict[str, any]]]:
    """複数のファイルパターンをスキャン

    Args:
        file_patterns: スキャン対象のファイルパターンリスト

    Returns:
        {ファイルパス: 違反リスト}の辞書
    """
    results = {}

    for pattern in file_patterns:
        # Pathオブジェクトに変換してglobを使用
        pattern_path = Path(pattern)

        if pattern_path.is_file():
            # 単一ファイル
            files = [pattern_path]
        else:
            # グロブパターン
            parent = pattern_path.parent if pattern_path.parent.exists() else Path(".")
            files = parent.glob(pattern_path.name)

        for file_path in files:
            if file_path.suffix == ".py":
                violations = detect_shift_negative(file_path)

                # 許可された関数での使用を除外
                filtered_violations = [
                    v for v in violations if not is_allowed_function(v["function"], file_path)
                ]

                if filtered_violations:
                    results[str(file_path)] = filtered_violations

    return results


def main():
    """メイン処理

    スキャン対象ファイルを定義し、データリークを検出する

    Returns:
        0: データリークなし（成功）
        1: データリーク検出（失敗）
    """
    print("🔍 Starting data leak detection...")
    print()

    # スキャン対象ファイル
    file_patterns = [
        "user_data/strategies/two_tier_strategy.py",
        "user_data/strategies/primary/*.py",
        "user_data/strategies/utils/*.py",
    ]

    # スキャン実行
    results = scan_files(file_patterns)

    # 結果表示
    if not results:
        print("✅ No data leakage detected!")
        print()
        print("All checked files:")
        for pattern in file_patterns:
            print(f"  ✓ {pattern}")
        return 0

    # 違反が検出された場合
    print("❌ Data leakage detected!")
    print()

    has_violations = False

    for file_path, violations in results.items():
        print(f"📄 {file_path}:")
        for v in violations:
            print(f"  Line {v['line']}: .shift(-n) in {v['function']}()")
            has_violations = True
        print()

    if has_violations:
        print("⚠️  Data leakage prevention guidelines:")
        print("  - populate_indicators() should NOT use .shift(-n)")
        print("  - Only calculate_returns() may use .shift(-n) for label generation")
        print("  - Future data must be isolated from features")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
