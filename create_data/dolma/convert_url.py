OUT_DIRECTORY = "/data/dolma/data"

url_list = []
f = open("dolma_v1_6.txt", 'r')
for line in f:
    # print(line)
    # import pdb; pdb.set_trace()
    folder_name = line.split('/')[-2]
    file_name = line.split('/')[-1]
    aria2_url =  f"{line} dir={OUT_DIRECTORY}/{folder_name}\n out={file_name}"
    url_list.append(aria2_url)
    
out = open("dolma_v1_6_aria2.txt", "a")
# Print the combined list of URLs
for i, url in enumerate(url_list):
    out.write(url + "\n")
