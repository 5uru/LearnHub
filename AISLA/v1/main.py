from fastapi import FastAPI
from pydantic import BaseModel

from AISLA.v1.translation import main as translator


class TranslationRequest(BaseModel):
    text: str
    src_lang: str = "dyu_Latn"
    tgt_lang: str = "fra_Latn"
    by_sentence: bool = True


app = FastAPI()


@app.post("/translate")
def translate(request: TranslationRequest):
    """
    Perform translation with a fine-tuned NLLB model.
    The language codes are supposed to be in 8-letter format, like "eng_Latn".
    Their list can be returned by /list-languages.
    """
    output = translator(
        request.text,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        by_sentence=request.by_sentence,
    )
    return {"translation": output}
