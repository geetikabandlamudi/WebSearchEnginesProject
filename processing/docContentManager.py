def fetchContentWithDocID(docid):
    docName = int(docid/5000)+1
    i = docid%5000

    fileName = "./bert_docs/intermediate_postings"+str(docName)+".txt"
    doc_content = ""

    with open(fileName, 'r', encoding='utf-8') as file:
        content = file.read()

        sections = content.split('\n\n\n')
        count = 1
        for section in sections:
            if(count==i):
                lines = section.split('\n')

                if len(lines) >= 2:
                    doc_content = ' '.join(lines[1:]).strip()  
            count = count+1
    return doc_content

# docid = 1001234
# content = fetchContentWithDocID(docid)
# print(content)
