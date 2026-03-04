from pipeline import DisinformationPipeline
import json

def main():
    print("Initializing Disinformation Pipeline...\n")
    pipeline = DisinformationPipeline()

    # Sample 1: A potentially fake news article
    sample_post_1 = {
        "headline": "¡INCREÍBLE! El secreto que los políticos no quieren que sepas sobre el nuevo virus mortal",
        "content": "Un nuevo estudio demuestra que el virus se esconde en el agua potable. Comparte esto antes de que lo borren. Es indignante la corrupción.",
        "source_url": "https://clickbait-news.net/article/123",
        "reported_date": "2020-05-15"  # Old date
    }

    # Sample 2: A likely legitimate news article
    sample_post_2 = {
        "headline": "Nuevas medidas económicas anunciadas para el próximo trimestre",
        "content": "El ministerio de economía ha publicado hoy el nuevo conjunto de medidas que afectarán a las pymes.",
        "source_url": "https://www.reuters.com/news/123",
        "reported_date": "2023-10-25" # Recent date relative to some static past
    }

    # Dynamic fix for sample 2: use today's date for 'recent' logic
    from datetime import datetime
    sample_post_2["reported_date"] = datetime.now().strftime("%Y-%m-%d")

    print("--- Analyzing Sample Post 1 (Suspicious) ---")
    print(f"Headline: {sample_post_1['headline']}")
    report_1 = pipeline.analyze_post(sample_post_1)
    pipeline.print_report(report_1)

    print("\n" + "="*50 + "\n")

    print("--- Analyzing Sample Post 2 (Legitimate) ---")
    print(f"Headline: {sample_post_2['headline']}")
    report_2 = pipeline.analyze_post(sample_post_2)
    pipeline.print_report(report_2)

if __name__ == "__main__":
    main()
