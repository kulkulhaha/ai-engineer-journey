"""Day3 함수 실험용 테스트. 풀이 구현은 포함하지 않는다.

노트북에서 이 파일을 %run으로 읽은 뒤
run_least_squares_checks(fit_least_squares)를 실행한다.
"""

import numpy as np


# 각 튜플: 이름, X(행=사례, 열=특성), y, 기대 계수, 기대 예측값
CASES = [
    ("기본 직선", [[0], [1], [2]], [1, 3, 5], [1, 2], [1, 3, 5]),
    ("음수 기울기", [[-2], [0], [2]], [7, 3, -1], [3, -2], [7, 3, -1]),
    ("0인 목표", [[-1], [0], [1]], [0, 0, 0], [0, 0], [0, 0, 0]),
    ("잡음 포함", [[0], [1], [2]], [1, 2, 5], [2 / 3, 2],
     [2 / 3, 8 / 3, 14 / 3]),
    ("특성 두 개", [[0, 0], [1, 0], [0, 1], [1, 1]], [1, 3, -2, 0],
     [1, 2, -3], [1, 3, -2, 0]),
    ("중복 특성: 최소 노름 계수", [[0, 0], [1, 1], [2, 2]], [1, 3, 5],
     [1, 1, 1], [1, 3, 5]),
    ("상수 특성: 최소 노름 계수", [[1], [1], [1]], [1, 2, 3],
     [1, 1], [2, 2, 2]),
    ("관측 한 개: 최소 노름 계수", [[1]], [2], [1, 1], [2]),
]


def run_least_squares_checks(fit, *, include_invalid=True):
    """개별 PASS/FAIL을 출력하고 결과 목록을 반환한다. 허용 오차는 1e-8.

    include_invalid=False로 정상 입력부터 실험할 수 있다.
    오류 입력은 ValueError 또는 TypeError를 명시적으로 발생시키는지 확인한다.
    """
    results = []

    def check(name, action):
        try:
            action()
        except Exception as exc:
            results.append((name, False))
            print(f"FAIL | {name} | {type(exc).__name__}: {exc}")
        else:
            results.append((name, True))
            print(f"PASS | {name}")

    def check_case(x, y, expected_w, expected_prediction, dtype=float):
        x, y = np.array(x, dtype=dtype), np.array(y, dtype=dtype)
        w = fit(x, y)
        assert isinstance(w, np.ndarray), "반환값은 np.ndarray여야 합니다."
        assert w.shape == (x.shape[1] + 1,), f"계수 형태 확인: {w.shape}"
        np.testing.assert_allclose(w, expected_w, atol=1e-8, rtol=1e-8)
        prediction = w[0] + x @ w[1:]
        np.testing.assert_allclose(prediction, expected_prediction, atol=1e-8, rtol=1e-8)

    for name, x, y, w, prediction in CASES:
        check(name, lambda: check_case(x, y, w, prediction))
    check("정수 입력에서도 소수 계수", lambda: check_case(
        [[0], [1], [2]], [1, 2, 5], [2 / 3, 2], [2 / 3, 8 / 3, 14 / 3], int))

    def compare_sklearn():
        from sklearn.linear_model import LinearRegression

        x = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, -1]], dtype=float)
        y = np.array([1, 3.2, -2.1, 0.3, 8.2])
        reference = LinearRegression(fit_intercept=True).fit(x, y)
        w = fit(x, y)
        np.testing.assert_allclose(w, np.r_[reference.intercept_, reference.coef_],
                                   atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(w[0] + x @ w[1:], reference.predict(x),
                                   atol=1e-8, rtol=1e-8)

    check("sklearn 대조: 독립 특성·잡음 포함", compare_sklearn)

    if include_invalid:
        invalid = [
            ("행 수 불일치", [[0], [1]], [1]),
            ("빈 관측", np.empty((0, 1)), []),
            ("X가 1차원", [0, 1, 2], [1, 3, 5]),
            ("X가 3차원", [[[0]], [[1]]], [1, 3]),
            ("y가 2차원", [[0], [1]], [[1], [3]]),
            ("y가 스칼라", [[0]], 1),
        ]
        for label, bad in [("NaN", np.nan), ("양의 무한대", np.inf),
                           ("음의 무한대", -np.inf)]:
            invalid.append((f"X에 {label}", [[0], [bad]], [1, 3]))
            invalid.append((f"y에 {label}", [[0], [1]], [1, bad]))

        def rejects(x, y):
            try:
                fit(np.array(x, dtype=float), np.array(y, dtype=float))
            except (ValueError, TypeError):
                return
            raise AssertionError("잘못된 입력을 오류로 알리지 않았습니다.")

        for name, x, y in invalid:
            check(name, lambda: rejects(x, y))

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} 통과")
    return results
