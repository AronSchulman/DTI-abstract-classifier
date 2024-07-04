import pandas as pd
from metapub import PubMedFetcher
import yaml

def main():
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    pmids = cfg["pmids"]
    pmids = [str(i) for i in pmids.split(',')]
    
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
    
    df_final.to_csv(cfg["metadata_save_dir"], index=False)
    
main()
