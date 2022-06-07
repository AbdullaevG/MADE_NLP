import torch

def flatten(l):
    return [item for sublist in l for item in sublist]

def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    output = model(src, trg, 0) #turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = get_text(list(trg[:,0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[1:, 0]), TRG_vocab)
    
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = elapsed_time / 60
    return elapsed_mins, elapsed_time


def translate_sentence(model, sentence, src_vocab, trg_vocab, device, max_len=100):
    model.eval()
    with torch.no_grad():
        
        src_enc = model.encoder(sentence)
        # src_enc = [src_len, 1, hid_dim]
        
        trg = [trg_vocab.stoi["<sos>"]]
        
        for _ in range(max_len):
            output = model.decoder(
                sentence, 
                src_enc, 
                torch.tensor(trg).unsqueeze(1).to(device),
            )
            # output = [tgt len, 1, tgt_vocab]
            output_idx = output.argmax(dim=-1)[-1, :].item()
            
            trg.append(output_idx)
            if output_idx == trg_vocab.stoi["<eos>"]:
                break
            
        return trg[1:]
    
    
