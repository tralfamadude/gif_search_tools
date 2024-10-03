Utilities and batch jobs to support GIF semantic search. The main jobs are json2img_embedding_standalone.py and  the Hadoop streaming version of it, clip_embedding_mapper.py. See the comments at the top of these files for details. 

This was used to support GIF semantic search at Internet Archive. 

The json2img_embedding_standalone.py job can apply 3 neural networks on millions of GIFs. OpenCLIP is used to get embeddings, Falsonsai/nsfw is used for image->nsfw rating, and BLIP2 is used to get keywords which can be used for augemting search and as another nsfw metric if specific naughty words are present.

In practice, we did not use BLIP2 because it processes images on our Nvidia L4 GPU at the rate of 1.7/sec; that is the long pole in the tent since OpenCLIP works around 75 images/sec. We ran this on 1.5M GIFs on 3 GPUs for about 24hr; if we used BLIIP2, it would take a month or more even when run with 8bit weights and fp16. If the 2.7b BLIP2 model were distilled into a smaller model, perhaps it would work nearly as well and be much faster.


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
| most_different_vectors_test.py      | Comprehensive test of most different vectors, used this for production |
| query_to_vector.py                  | QueryEmbedding class to convert a query to an embedding      |
| select_representative_vectors_test.py| Another most different vectors test using KMeans            |
| topk_vectors_test.py                | Find representative vectors with KMeans, scipy cosine        |
| verify_clip_model_load.py           | Utility to verify that OpenCLIP model in local fs is good    |
| run.sh                              | How I ran the standalone job                                 |

