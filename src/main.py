import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import SpamDataLoader
from classifiers import SpamClassifier, LLMClassifier
from metrics import MetricsCalculator

def main():
    print("=" * 60)
    print("СИСТЕМА РАСПОЗНАВАНИЯ СООБЩЕНИЙ ОТ БОТОВ")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n[1] Загрузка данных...")
    loader = SpamDataLoader()
    messages, labels = loader.create_sample_dataset()
    loader.save_to_csv()
    print(f"Загружено {len(messages)} сообщений")
    
    # 2. Тестирование LLM-подхода (Qwen3.5-Plus)
    print("\n[2] Тестирование LLM-классификатора (Qwen3.5-Plus)...")
    classifier_llm = LLMClassifier()
    predictions_llm = classifier_llm.predict(messages)
    metrics_llm = MetricsCalculator.calculate_metrics(labels, predictions_llm)
    
    print(f"Precision: {metrics_llm['precision']:.4f}")
    print(f"Recall: {metrics_llm['recall']:.4f}")
    print(f"F1-score: {metrics_llm['f1_score']:.4f}")
    
    # 3. Тестирование TF-IDF + SVM
    print("\n[3] Обучение модели TF-IDF + SVM...")
    classifier_svm = SpamClassifier()
    classifier_svm.train(messages, labels)
    predictions_svm = classifier_svm.predict(messages)
    metrics_svm = classifier_svm.evaluate(labels, predictions_svm)
    
    print(f"Precision: {metrics_svm['precision']:.4f}")
    print(f"Recall: {metrics_svm['recall']:.4f}")
    print(f"F1-score: {metrics_svm['f1']:.4f}")
    
    # 4. Сохранение результатов
    print("\n[4] Сохранение результатов...")
    MetricsCalculator.save_results({
        'tfidf_svm': metrics_svm,
        'llm_qwen': metrics_llm
    })
    MetricsCalculator.plot_confusion_matrix(labels, predictions_llm)
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)

if __name__ == "__main__":
    main()