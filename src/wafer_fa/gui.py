from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from .config import AppConfig
from .image_io import imread_unicode
from .service import WaferFAService


class WaferFAGui:
    def __init__(self, root: tk.Tk, service: WaferFAService):
        self.root = root
        self.service = service
        self.root.title("Wafer Particle Pattern Historical Case Retrieval / FA Assistant")
        self.root.geometry("1320x820")
        self.root.minsize(1080, 700)
        self._images: list[ImageTk.PhotoImage] = []
        self.query_photo: ImageTk.PhotoImage | None = None
        self.query_path: Path | None = None

        self._build()
        self._refresh_status()

    def _build(self) -> None:
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="选择查询图像", command=self.choose_query).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="检索 Top 3", command=self.run_search).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="导入历史案例", command=self.add_case).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="重建特征", command=self.rebuild).pack(side=tk.LEFT, padx=(0, 8))

        self.status = ttk.Label(toolbar, text="")
        self.status.pack(side=tk.RIGHT)

        hint = ttk.Label(
            self.root,
            text="结果百分比 = 相似度评分，不是分类概率/置信度。缩放通过 wafer 圆归一化处理。",
            padding=(10, 0, 10, 8),
        )
        hint.pack(fill=tk.X)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left = ttk.Labelframe(body, text="查询图像", padding=10)
        right = ttk.Labelframe(body, text="Top 3 历史相似案例", padding=10)
        body.add(left, weight=1)
        body.add(right, weight=3)

        self.query_image_label = ttk.Label(left, text="尚未选择图像", anchor=tk.CENTER)
        self.query_image_label.pack(fill=tk.BOTH, expand=True)
        self.query_info = ttk.Label(left, text="", justify=tk.LEFT, wraplength=300)
        self.query_info.pack(fill=tk.X, pady=(10, 0))

        canvas = tk.Canvas(right, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=canvas.yview)
        self.results_frame = ttk.Frame(canvas)
        self.results_frame.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _refresh_status(self) -> None:
        self.status.config(text=f"历史案例：{self.service.db.count()} 张")

    def _photo(self, path: str | Path, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
        image = imread_unicode(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._images.append(photo)
        return photo

    def choose_query(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 wafer particle map",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        self.query_path = Path(path)
        try:
            photo = self._photo(self.query_path, (360, 520))
            self.query_photo = photo
            self.query_image_label.config(image=photo, text="")
            features, normalized = self.service.analyze(self.query_path)
            self.query_info.config(
                text=(
                    f"文件：{self.query_path.name}\n"
                    f"Particle：{features.particle_count}\n"
                    f"Wafer 检测：{normalized.geometry.detection_method}\n"
                    f"圆心：({normalized.geometry.center_x:.1f}, {normalized.geometry.center_y:.1f})\n"
                    f"半径：{normalized.geometry.radius:.1f}px"
                )
            )
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))

    def add_case(self) -> None:
        path = filedialog.askopenfilename(
            title="选择要加入历史库的 particle map",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        comment = simpledialog.askstring("历史 Comment", "请输入该案例当时的 comment / FA 结论：", parent=self.root)
        if comment is None:
            return
        meta_text = simpledialog.askstring(
            "Metadata（可选）",
            "可输入 JSON，例如：\n{\"tool\":\"OX-01\",\"chamber\":\"C3\",\"recipe\":\"RCP-A\"}",
            parent=self.root,
        )
        metadata = {}
        if meta_text:
            try:
                metadata = json.loads(meta_text)
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be a JSON object")
            except Exception as exc:
                messagebox.showerror("Metadata 格式错误", str(exc))
                return
        try:
            case_id = self.service.add_case(path, comment, metadata)
            self._refresh_status()
            messagebox.showinfo("导入完成", f"Case #{case_id} 已加入历史库")
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def _clear_results(self) -> None:
        for child in self.results_frame.winfo_children():
            child.destroy()
        self._images = [self.query_photo] if self.query_photo is not None else []

    def run_search(self) -> None:
        if self.query_path is None:
            self.choose_query()
            if self.query_path is None:
                return
        if self.service.db.count() == 0:
            messagebox.showwarning("历史库为空", "请先导入历史案例，或运行 CLI demo 生成演示数据。")
            return
        try:
            results = self.service.search(self.query_path, top_k=3)
        except Exception as exc:
            messagebox.showerror("检索失败", str(exc))
            return

        self._clear_results()
        for rank, result in enumerate(results, 1):
            card = ttk.Labelframe(self.results_frame, text=f"#{rank}  Case {result.case.id}", padding=10)
            card.pack(fill=tk.X, expand=True, pady=(0, 10))

            image_holder = ttk.Label(card)
            image_holder.grid(row=0, column=0, rowspan=5, sticky="nw", padx=(0, 12))
            try:
                photo = self._photo(result.case.image_path, (260, 260))
                image_holder.config(image=photo)
            except Exception:
                image_holder.config(text="图片不可读")

            score = ttk.Label(card, text=f"相似度：{result.score * 100:.1f}%", font=("Segoe UI", 14, "bold"))
            score.grid(row=0, column=1, sticky="w")

            ttk.Label(card, text="Comment：", font=("Segoe UI", 10, "bold")).grid(row=1, column=1, sticky="nw")
            ttk.Label(card, text=result.case.comment or "(空)", wraplength=650, justify=tk.LEFT).grid(
                row=2, column=1, sticky="nw"
            )

            meta = json.dumps(result.case.metadata, ensure_ascii=False) if result.case.metadata else "{}"
            ttk.Label(card, text=f"Metadata：{meta}", wraplength=650, justify=tk.LEFT).grid(
                row=3, column=1, sticky="nw", pady=(6, 0)
            )

            comp = "  ".join(f"{k}={v*100:.0f}%" for k, v in result.components.items())
            ttk.Label(card, text=f"分项：{comp}", wraplength=650, justify=tk.LEFT).grid(
                row=4, column=1, sticky="nw", pady=(6, 0)
            )
            card.columnconfigure(1, weight=1)

    def rebuild(self) -> None:
        if not messagebox.askyesno("重建特征", "将重新读取所有历史图片并重算特征。继续？"):
            return
        stats = self.service.rebuild()
        messagebox.showinfo("完成", f"成功：{stats['rebuilt']}\n失败：{stats['failed']}")


def run_gui() -> None:
    root = tk.Tk()
    service = WaferFAService(AppConfig(root=Path.cwd()))
    WaferFAGui(root, service)
    root.mainloop()
