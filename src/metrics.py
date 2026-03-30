import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(true_labels, pred_labels):
        """Расчет всех метрик качества"""
        tp = sum((t == 1 and p == 1) for t, p in zip(true_labels, pred_labels))
        fp = sum((t == 0 and p == 1) for t, p in zip(true_labels, pred_labels))
        fn = sum((t == 1 and p == 0) for t, p in zip(true_labels, pred_labels))
        tn = sum((t == 0 and p == 0) for t, p in zip(true_labels, pred_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(true_labels)
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'accuracy': round(accuracy, 4),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    
    @staticmethod
    def plot_confusion_matrix(true_labels, pred_labels, save_path='results/confusion_matrix.png'):
        """Визуализация матрицы ошибок"""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Не спам', 'Спам'],
                    yticklabels=['Не спам', 'Спам'])
        plt.title('Матрица ошибок классификации')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Матрица ошибок сохранена в {save_path}")
    
    @staticmethod
    def save_results(metrics, filepath='results/metrics.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)