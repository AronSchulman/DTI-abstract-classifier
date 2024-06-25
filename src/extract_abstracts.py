import pandas as pd
from metapub import PubMedFetcher


def main():
    pmids = [35598299, 37094465, 37244151, 32142454] # list your PubMed IDs here
    pmids = [str(i) for i in pmids]

    fetch = PubMedFetcher()
    data = []
    for pmid in pmids:
        try:
            article = fetch.article_by_pmid(pmid)
            title = article.title
            abstract = article.abstract
            journal = article.journal
            year = article.year
            if title and abstract:
                data.append((pmid, title + " " + abstract, journal, year))
            else:
                print("No title or abstract for PMID", pmid)
        except:
            print("Error caused by PMID", pmid)
    df_final = pd.DataFrame(data, columns=["PMID", "text", "journal", "year"])
    df_final.to_csv(f"{len(pmids)}_abstracts.csv", index=False)

main()