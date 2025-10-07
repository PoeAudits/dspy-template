import dspy

def get_prompt(optimized_program):
    prompt = { 
        name: dspy.ChatAdapter().format(
            p.signature,
            demos=p.demos,
            inputs={k: f"{{{k}}}" for k in p.signature.input_fields},
            )
            for name, p in optimized_program.named_predictors()
            }['self']
    return prompt
                     
