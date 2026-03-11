import numpy as np


VOCAB_SIZE  = 10_000
D_MODEL     = 512
MAX_STEPS   = 50    


np.random.seed(0)
VOCAB = [f"palavra_{i}" for i in range(VOCAB_SIZE)]
VOCAB[0]    = "<PAD>"
VOCAB[1]    = "<START>"
VOCAB[2]    = "<EOS>"

VOCAB[3]    = "o"
VOCAB[4]    = "rato"
VOCAB[5]    = "roeu"
VOCAB[6]    = "a"
VOCAB[7]    = "roupa"
VOCAB[8]    = "do"
VOCAB[9]    = "rei"

ID_TO_WORD = {i: w for i, w in enumerate(VOCAB)}


def softmax(x):
    """Softmax numericamente estavel."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def mock_decoder_forward(current_sequence, encoder_out, W_proj):
    seq_len = len(current_sequence)

    token_vectors = np.array([
        np.sin(np.arange(D_MODEL) * (tok_id + 1) * 0.01)
        for tok_id in current_sequence
    ])  

    
    decoder_hidden = token_vectors.mean(axis=0)  

    encoder_context = encoder_out[0].mean(axis=0)  
    combined = decoder_hidden + encoder_context    

    logits = combined @ W_proj  

    eos_bias = (seq_len - 1) * 2.5  
    logits[2] += eos_bias            

    probs = softmax(logits)

    return probs


def generate_next_token(current_sequence, encoder_out, W_proj):
    probs = mock_decoder_forward(current_sequence, encoder_out, W_proj)
    next_token_id = int(np.argmax(probs))
    return next_token_id, probs



def autoregressive_loop(encoder_out, W_proj, max_steps=MAX_STEPS):
    START_ID = VOCAB.index("<START>")
    EOS_ID   = VOCAB.index("<EOS>")

    generated_ids = [START_ID]

    print("=" * 50)
    print("LOOP DE INFERENCIA AUTO-REGRESSIVO")
    print("=" * 50)
    print(f"Token inicial: {ID_TO_WORD[START_ID]}")
    print()

    step = 0
    while step < max_steps:
        next_id, probs = generate_next_token(generated_ids, encoder_out, W_proj)

        palavra = ID_TO_WORD[next_id]
        print(f"Passo {step + 1:02d} | token gerado: [{next_id:5d}] '{palavra}'"
              f"  (prob: {probs[next_id]:.4f})")

        generated_ids.append(next_id)

        if next_id == EOS_ID:
            print()
            print("[EOS detectado] Geracao encerrada.")
            break

        step += 1
    else:
        print()
        print("[AVISO] Limite maximo de passos atingido sem <EOS>.")

    frase_ids = [i for i in generated_ids if i not in (START_ID, EOS_ID)]
    frase_final = " ".join(ID_TO_WORD[i] for i in frase_ids)

    print()
    print("=" * 50)
    print(f"FRASE GERADA: \"{frase_final}\"")
    print("=" * 50)

    return frase_final


# ---------- Execucao ----------

if __name__ == "__main__":
    np.random.seed(99)

    encoder_out = np.random.randn(1, 10, D_MODEL)

    W_proj = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.05

    W_proj[:, 3]  *= 20 
    W_proj[:, 4]  *= 18   
    W_proj[:, 5]  *= 16   
    W_proj[:, 6]  *= 14
    W_proj[:, 7]  *= 12    
    W_proj[:, 8]  *= 10 
    W_proj[:, 9]  *= 8     
    W_proj[:, 2]  *= 6    

    frase = autoregressive_loop(encoder_out, W_proj)