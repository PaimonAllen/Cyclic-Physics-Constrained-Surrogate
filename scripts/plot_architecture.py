"""Render the implemented two-branch architecture in English and Chinese."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]


def draw(language):
    zh = language == "zh"
    font = "Microsoft YaHei" if zh else "Arial"
    plt.rcParams.update(
        {
            "font.family": font,
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="#FFFFFF")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    colors = {
        "ink": "#172D42",
        "muted": "#586D80",
        "blue": "#EAF3FC",
        "green": "#E8F5EF",
        "orange": "#FFF2E3",
    }

    def text(x, y, s, size=12, **kw):
        ax.text(
            x, y, s, fontsize=size, color=colors["ink"], ha="center", va="center", **kw
        )

    def box(x, y, w, h, s, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=color,
                edgecolor="#ADC1D1",
                linewidth=1,
            )
        )
        text(x + w / 2, y + h / 2, s, 12, linespacing=1.7)

    def arrow(a, b, style="-", rad=0):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=15,
                color="#53728B",
                linewidth=1.4,
                linestyle=style,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    title = (
        "物理约束循环响应代理网络"
        if zh
        else "Physics-constrained cyclic response surrogate"
    )
    text(0.5, 0.945, title, 24, fontweight="bold")
    text(
        0.5,
        0.885,
        (
            "12 个导轨  ·  6 个局部特征  ·  29 个全局特征  ·  隐层宽度 256"
            if zh
            else "12 rails  ·  6 local features  ·  29 global features  ·  latent width 256"
        ),
        13,
    )
    box(
        0.02,
        0.53,
        0.16,
        0.26,
        (
            "部署命令与查询时间\n局部输入 12 × 6\n全局上下文 29\n训练集统计标准化"
            if zh
            else "Command + query time\nLocal inputs 12 × 6\nGlobal context 29\nTraining-only normalization"
        ),
        colors["blue"],
    )
    box(
        0.235,
        0.68,
        0.15,
        0.14,
        (
            "共享局部编码器\n全局编码与广播"
            if zh
            else "Shared local encoder\nGlobal encoder + broadcast"
        ),
        colors["blue"],
    )
    box(
        0.445,
        0.68,
        0.16,
        0.14,
        (
            "6 个循环残差块\n卷积核 3 · 循环填充"
            if zh
            else "6 circular residual blocks\nKernel 3 · circular padding"
        ),
        colors["blue"],
    )
    box(
        0.66,
        0.68,
        0.12,
        0.14,
        (
            "逐导轨输出头\n位移 · 速度先验\n加速度 · 推力残差"
            if zh
            else "Railwise heads\nΔp · velocity prior\na · thrust residual"
        ),
        colors["blue"],
    )
    box(
        0.835,
        0.54,
        0.145,
        0.28,
        (
            "反标准化与载荷恢复\n推力饱和裁剪\nPD 等式重构速度\nR = u − m a\n输出 p, v, a, R, u"
            if zh
            else "Restore physical scales\nClip actuator thrust\nReconstruct velocity by PD\nR = u − m a\nOutput p, v, a, R, u"
        ),
        colors["orange"],
    )
    arrow((0.18, 0.75), (0.227, 0.75))
    arrow((0.393, 0.75), (0.437, 0.75))
    arrow((0.613, 0.75), (0.652, 0.75))
    arrow((0.788, 0.75), (0.827, 0.75))
    box(
        0.235,
        0.205,
        0.15,
        0.14,
        (
            "独立能量编码器\n共享导轨参数"
            if zh
            else "Separate energy encoders\nShared rail parameters"
        ),
        colors["green"],
    )
    box(
        0.445,
        0.205,
        0.16,
        0.14,
        (
            "状态特征注入\n3 个循环残差块"
            if zh
            else "Response-latent injection\n3 circular residual blocks"
        ),
        colors["green"],
    )
    box(
        0.66,
        0.205,
        0.12,
        0.14,
        (
            "循环不变汇聚\n均值 · 最大值 · RMS"
            if zh
            else "Invariant pooling\nMean · max · RMS"
        ),
        colors["green"],
    )
    box(
        0.835,
        0.205,
        0.145,
        0.14,
        (
            "能量 MLP\n全局壳体应变能 U"
            if zh
            else "Energy MLP\nGlobal shell strain energy U"
        ),
        colors["green"],
    )
    arrow((0.12, 0.525), (0.227, 0.275), rad=0.18)
    arrow((0.393, 0.275), (0.437, 0.275))
    arrow((0.613, 0.275), (0.652, 0.275))
    arrow((0.788, 0.275), (0.827, 0.275))
    arrow((0.715, 0.672), (0.53, 0.353), style="--", rad=0.1)
    text(0.39, 0.48, "响应隐变量" if zh else "Response latents", 11)
    text(0.31, 0.61, "响应分支" if zh else "Response branch", 13, fontweight="bold")
    text(0.31, 0.15, "能量分支" if zh else "Energy branch", 13, fontweight="bold")
    text(
        0.5,
        0.065,
        (
            "硬 PD 关系对应未饱和等式，时间导数一致性和饱和工况通过独立诊断评估。"
            if zh
            else "The hard PD relation represents the unsaturated equality. Temporal and saturated-case diagnostics are evaluated separately."
        ),
        11,
    )
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.01, top=0.99)
    out = ROOT / "docs/figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out / f"architecture_{language}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw("en")
    draw("zh")
