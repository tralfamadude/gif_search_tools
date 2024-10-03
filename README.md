Utilities and batch jobs to support GIF semantic search. The main jobs are json2img_embedding_standalone.py and  the Hadoop streaming version of it, clip_embedding_mapper.py. See the comments at the top of these files for details. 

This was used to support GIF semantic search at Internet Archive. 



| File                                | Purpose                                                     |
|-------------------------------------|-------------------------------------------------------------|
| NSFW.py                             | Class for NSFW rating of images                              |
| clip_embedding_mapper.py            | Python streaming Hadoop job for image->embedding, keywords   |
| clip_to_local.py                    | Utility to copy a Hugging Face hub model to local file system|
| gif_exploder.py                     | Utility to extract images from a GIF file                    |
| image_embedding_latency.py          | Measurement latency of computing an embedding                |
| img2txt_blip2.py                    | BLIP2 class for image to caption                             |
| json2img_embedding_standalone.py    | The main job for image->embedding, keywords                  |
| jsonl_to_gifs.py                    | Utility to extract GIFs from jsonl                           |
| keyword_extractor.py                | Class to help find keywords from captions, remove stop words |
| most_different_vectors_test.py      | Test of using KMeans to find most different vectors          |
| query_to_vector.py                  | QueryEmbedding class to convert a query to an embedding      |
| select_representative_vectors_test.py| Another most different vectors test                         |
| similar_vector_set_test.py          | Check to see if all vectors in a list are similar            |
| similar_vectors_test.py             | Check to see if all vectors in a list are similar            |
| topk_vectors_test.py                | Find representative vectors with KMeans                      |
| verify_clip_model_load.py           | Utility to verify that OpenCLIP model in local fs is good    |

