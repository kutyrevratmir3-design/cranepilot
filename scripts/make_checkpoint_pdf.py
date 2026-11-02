"""Human-readable research checkpoint; this is not the compiled LaTeX paper."""
from pathlib import Path
import json
import pandas as pd
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak


def main():
    root=Path(__file__).resolve().parents[1]
    result=root/"results/latest"
    summary=json.loads((result/"summary.json").read_text())
    font_root=Path("C:/Windows/Fonts")
    regular,bold=font_root/"arial.ttf",font_root/"arialbd.ttf"
    if not regular.exists():
        from matplotlib import font_manager
        regular=Path(font_manager.findfont("DejaVu Sans"))
        bold=Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",weight="bold")))
    pdfmetrics.registerFont(TTFont("Research",str(regular)))
    pdfmetrics.registerFont(TTFont("ResearchBold",str(bold)))
    pdfmetrics.registerFontFamily("Research",normal="Research",bold="ResearchBold")
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ResearchTitle",fontName="ResearchBold",fontSize=22,leading=27,textColor=colors.HexColor("#143f52"),spaceAfter=10))
    styles.add(ParagraphStyle(name="ResearchHeading",fontName="ResearchBold",fontSize=13,leading=17,spaceBefore=7,spaceAfter=7,textColor=colors.HexColor("#143f52")))
    styles.add(ParagraphStyle(name="ResearchBody",fontName="Research",fontSize=9.5,leading=13.5,spaceAfter=7))
    styles.add(ParagraphStyle(name="ResearchSmall",fontName="Research",fontSize=8,leading=11,spaceAfter=5))
    story=[]
    def text(s,style="ResearchBody"):
        story.append(Paragraph(s,styles[style]))
    def graphic(name,width=169*mm):
        path=root/"paper/figures"/f"{name}.png"
        with PILImage.open(path) as p:
            w,h=p.size
        story.append(Image(str(path),width=width,height=width*h/w))
        story.append(Spacer(1,3*mm))
    def table(rows,widths):
        rows=[[Paragraph(str(c),styles["ResearchSmall"]) for c in row] for row in rows]
        t=Table(rows,colWidths=widths,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7f0f4")),
                              ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#176b87")),
                              ("VALIGN",(0,0),(-1,-1),"TOP"),
                              ("BOTTOMPADDING",(0,0),(-1,-1),5),
                              ("TOPPADDING",(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,4*mm))
    text("CranePilot","ResearchTitle")
    text("Первый рабочий этап исследования · 2 ноября 2026 года","ResearchHeading")
    text("Python, экспериментальный прогноз и отдельное управление в симуляции. Основа - прежняя работа Ратмира Кутырева и существующие измерения Inteco. Новые физические эксперименты не требуются для текущего объёма работы.")
    table([["Проверенный результат","Объём"],["Исходные записи; совпадение контрольных сумм","33 из 33"],
           ["Повторение прежней линейной модели","24 значения; расхождение около 1e-13°"],
           ["Сценарии и оси в общем benchmark","38 сочетаний"],
           ["Обученные сети в общем сравнении",str(summary["trained_neural_networks"])],
           ["Управление в симуляции","16 парных условий; 32 эпизода"],
           ["Пройденные научные и численные тесты","13"]],[125*mm,44*mm])
    text("В первом тесте переноса калиброванная физика даёт 0,758° MAE на горизонте 1 с. Проекционная ML-поправка даёт 0,907°: общее превосходство нового кандидата пока не подтверждено.")
    graphic("forecast_comparison",154*mm)
    story.append(PageBreak())
    text("Что показал прогноз","ResearchTitle")
    text("Первый тест: обучение на сценариях 3, 5, 6; валидация на 4; оценка на 1, 2, 7, 8. У каждого метода одинаковая доступная история из 41 углового измерения. Будущие измерения не входят в признаки. Это ретроспективная оценка уже изученного датасета.")
    graphic("horizon_comparison")
    text("Результат зависит от оси движения","ResearchHeading")
    ns=pd.read_csv(result/"all_family_neural_summary.csv")
    q=ns[ns.horizon_s==1].pivot(index=["family","axis"],columns="model",values="mae_deg")
    rows=[["Семейство / ось","Физика","GP residual","Проекционная NN"]]
    for (family,axis),r in q.iterrows():
        rows.append([family+" / "+axis,*[f"{r[m]:.3f}°" for m in ["calibrated_oscillator","gp_residual","projected_phase_residual"]]])
    table(rows,[65*mm,30*mm,35*mm,39*mm])
    text("На продольной раскачке физическая модель остаётся очень сильной. Для некоторых поперечных движений обучаемые поправки полезны, но GP - серьёзный конкурент. Эти результаты не позволяют объявить проекционную сеть новым лучшим методом.")
    text("Оценки усреднены по целым сценариям. Перекрывающиеся временные окна и разные seeds не считаются независимыми физическими опытами.","ResearchSmall")
    story.append(PageBreak())
    text("Управление и адаптация","ResearchTitle")
    text("Симуляционный MPC использует нелинейный объект с задержкой привода, dead-zone и несовпадением параметров. Это отдельный пилот с доступным полным состоянием; нейросетевой прогноз пока не используется внутри контроллера.")
    graphic("control_paired",163*mm)
    text("Средняя остаточная RMS-раскачка: 0,0805° у nominal MPC и 0,0639° у calibrated MPC. Среднее время установления: 2,99 и 2,98 с. Все 32 выбранных эпизода завершились без отказов решателя. Это результат конкретных синтетических условий, не аппаратная проверка.","ResearchSmall")
    graphic("adaptation_curve",151*mm)
    text("Простая адаптация по начальному участку ухудшила результат. Все бюджеты оценены на одном и том же позднем интервале. Следующая задача - надёжность оценки параметров и правило отказа от плохой локальной адаптации.","ResearchSmall")
    text("Статус: рабочий исследовательский черновик. Доказательство новизны, полный action-conditioned метод, независимая симуляционная проверка и финальная журнальная версия остаются следующими этапами.","ResearchSmall")

    path=result/"CranePilot_Checkpoint_2026-11-02.pdf"
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,topMargin=17*mm,bottomMargin=17*mm)
    def footer(canvas,document):
        canvas.setFont("Research",8);canvas.setFillColor(colors.HexColor("#63727a"))
        canvas.drawString(20*mm,10*mm,"CranePilot · исследовательский checkpoint")
        canvas.drawRightString(A4[0]-20*mm,10*mm,str(document.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    import pypdfium2 as pdfium
    pdf=pdfium.PdfDocument(str(path))
    qa=root/".local/qa";qa.mkdir(parents=True,exist_ok=True)
    for j in range(len(pdf)):
        pdf[j].render(scale=1.4).to_pil().save(qa/f"checkpoint_{j+1}.png")
    print(json.dumps({"path":str(path),"pages":len(pdf),"qa":str(qa)}))


if __name__=="__main__":
    main()

