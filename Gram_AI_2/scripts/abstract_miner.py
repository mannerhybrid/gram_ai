import os
import pandas as pd
from Bio import Entrez
from bs4 import BeautifulSoup as soup


def get_Idlist(term="Adiposity"):
    Entrez.email = "md.nur.hakim.rosli@gmail.com"

    # 1. eSearch for a list of terms

    search_handle = Entrez.esearch(db="pubmed",
                                   term="adiposity",
                                   sort="relevance",
                                   retmode="xml",
                                   retmax=100000,
                                   usehistory="y")
    search_record = Entrez.read(search_handle)
    # record_count = search_record['Count']
    # search_handle = Entrez.esearch(db="pubmed",
    #                                term="adiposity",
    #                                sort="relevance",
    #                                retmode="xml",
    #                                usehistory="y",
    #                                retmax=record_count)
    # search_record = Entrez.read(search_handle)
    webenv = search_record['WebEnv']
    query_key = search_record['QueryKey']
    idlist = search_record['IdList']
    idstring = ",".join(idlist)
    return idlist

def miner(idlist):
    
    df = dict(
    year_published = [],
    title = [],
    abstract = [],
    authors = [],
    qualifiers = [],
    descriptors = [])
    
    for i in range(len(idlist)):
        # print(i)
        errant_ids = []
        disabled = 0
        passed = 0
        fetch_handle = Entrez.efetch("pubmed", id=idlist[i], rettype="xml", retmode="abstract")
        fetch_record = soup(fetch_handle.read(), "xml")
        article = fetch_record.PubmedArticleSet.PubmedArticle.MedlineCitation
        meshes = article.findAll("MeshHeading")
        authors = article.Article.AuthorList.findAll("Author")
        try:
            try:
                df['year_published'].append(article.DateCompleted.Year.text)
                df['title'].append(article.Article.ArticleTitle.text)
                df['abstract'].append(" ".join([section.text for section in article.Article.Abstract.findAll("AbstractText")]))

                passed += 1
                new_total = len(idlist) - disabled
                if i % 10 == 0:
                    print("[+] Successfully passed %d out of %d IDs." % (i, new_total))
                df['authors'].append([(idlist[i],
                                authors[a].Author.ForeName.text,
                                authors[a].Author.LastName.text) for a in range(len(authors))])
                df['qualifiers'].append([(idlist[i],
                                        meshes[a].QualifierName["UI"],
                                        meshes[a].QualifierName.text,
                                        meshes[a].DescriptorName["UI"],
                                        meshes[a].DescriptorName.text) for a in range(len(meshes))
                                        if meshes[a].find("QualifierName")])
                df['descriptors'].append([(idlist[i], "None",
                                "None",
                                meshes[a].DescriptorName["UI"],
                                meshes[a].DescriptorName.text) for a in range(len(meshes))
                            if not meshes[a].find("QualifierName")])
                # author_records.extend(author_list)
                # mesh_records.extend(descriptors)
                # mesh_records.extend(qualified_descriptors)
            except:
                disabled += 1
                errant_ids.append(id)
                continue
        except KeyboardInterrupt or AttributeError:
            df = pd.DataFrame(df)
            df.to_csv('..\\data\\abs_full.csv')
            print("[+] Completed parsing information from Pubmed!")


    
idlist = get_Idlist()
miner(idlist)