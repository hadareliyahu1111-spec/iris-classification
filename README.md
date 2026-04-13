# מסווג פרחי האירוס — Iris Flower Classifier

## תיאור הפרויקט

פרויקט זה בונה מסווג למידת מכונה לזיהוי מיני פרחי האירוס באמצעות רשת עצבית מלאכותית (MLPClassifier).
המודל מאומן על מערך הנתונים הקלאסי Iris מספריית `scikit-learn`, ומשיג דיוק של **100%** על קבוצת הבדיקה.

## שם קוד הצוות

**biu-he01**

## מבנה הקבצים

| קובץ | תיאור |
|------|--------|
| `model.py` | סקריפט Python המריץ את כל תהליך האימון וההערכה |
| `PRD.md` | מסמך דרישות המוצר (Product Requirements Document) |
| `PLAN.md` | תכנית הפיתוח — שלבי הבנייה של המסווג |
| `REPORT.md` | דוח מלא עם ניתוח תוצאות, מטריצת הבלבול ועקומת האובדן |
| `README.md` | קובץ זה — סקירה כללית של הפרויקט |
| `TODO.md` | רשימת משימות לפרויקט |
| `confusion_matrix.png` | גרף מטריצת הבלבול |
| `loss_curve.png` | גרף עקומת האובדן לאורך האימון |

## דרישות מוקדמות

```bash
pip install scikit-learn matplotlib seaborn numpy
```

## הרצת הקוד

```bash
python model.py
```

### פלט צפוי:

```
Training samples : 120
Testing  samples : 30
Training converged after 300 iterations.

Confusion Matrix:
[[10  0  0]
 [ 0 10  0]
 [ 0  0 10]]

Test Accuracy: 100.00%
Saved: confusion_matrix.png
Saved: loss_curve.png
```

הגרפים יישמרו אוטומטית בתיקיית הפרויקט.
