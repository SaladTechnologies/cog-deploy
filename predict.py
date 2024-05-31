from cog import BasePredictor, Input, Path
import time
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image


class Predictor(BasePredictor):
    def setup(self) -> None:
        
        start = time.perf_counter()
        print("Start to download and load the models.......")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

       # https://github.com/salesforce/LAVIS/blob/main/examples/blip_image_captioning.ipynb
       # base_coco, large_coco
        model1, vis_processors1, _                = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=self.device) 
 
        # https://github.com/salesforce/LAVIS/blob/main/examples/blip_vqa.ipynb
        model2, vis_processors2, text_processors2 = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device)
 
        # https://github.com/salesforce/LAVIS/blob/main/examples/blip_image_text_matching.ipynb
        # base, large
        model3, vis_processors3, text_processors3 = load_model_and_preprocess(
            name="blip_image_text_matching", model_type="base", is_eval=True, device=self.device)
        
        self.models = {
            "image_captioning":          [model1, vis_processors1, None            ],
            "visual_question_answering": [model2, vis_processors2, text_processors2],
            "image_text_matching":       [model3, vis_processors3, text_processors3]
        }

        end = time.perf_counter()
        print(f"The 3 models have been loaded in {round(end-start,1)} seconds.",flush=True)


    def predict(
        self,
        image: Path = Input( description="Input image" ),
        task: str = Input(
            choices=[ "image_captioning", "visual_question_answering", "image_text_matching" ],
            default="image_captioning", description="Choose a task." ),
        question: str = Input( default=None,
            description="Type question for the input image for visual question answering task." ),
        caption: str = Input( default=None,
            description="Type caption for the input image for image text matching task." ),
    ) -> str:

        start = time.perf_counter()

        if task == "visual_question_answering":
            assert ( question is not None ), "Please type a question for visual question answering task."
        if task == "image_text_matching":
            assert ( caption is not None ), "Please type a caption for image text matching task."

        model = self.models[task][0]
        vis_processors = self.models[task][1]
        text_processors = self.models[task][2]

        raw_image = Image.open(image).convert('RGB') # The file is already downloaded by the Cog Runner
                                                     # By default, PIL is agnostic about color spaces
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(self.device) # Tensor

        if task == "image_captioning":            
            caption = model.generate({"image": image})
            result = "Caption: " + caption[0]

        elif task == "visual_question_answering":
            question = text_processors["eval"](question)
            answer = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
            result = "Answer: " + answer[0]

        else: # image_text_matching
            text = text_processors["eval"](caption)

            itm_output = model( {"image": image, "text_input": text}, match_head="itm")
            itm_score = torch.nn.functional.softmax(itm_output, dim=1)
            itc_score = model( {"image": image, "text_input": text}, match_head='itc')
            result = (
            f"The image and text are matched with a probability of {itm_score[:, 1].item():.4%}.\n"
            f"The image feature and text feature has a cosine similarity of {itc_score.item():.4f}." )

        end = time.perf_counter()

        print(f"Inference time: {round(end-start,2)} seconds.")
        #print(result)

        return result

'''
2625148 -rw-rw-r-- 1 ubuntu ubuntu 2688144147 May 28 15:02 blip_coco_caption_base.pth
1864028 -rw-rw-r-- 1 ubuntu ubuntu 1908759703 May 28 16:33 model_base_retrieval_coco.pth      #
1412356 -rw-rw-r-- 1 ubuntu ubuntu 1446244375 May 28 15:03 model_base_vqa_capfilt_large.pth   #
'''