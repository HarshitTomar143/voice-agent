class MicProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0][0]
    if (input != null){
        console.log(input.length)
    }
    return true

  }
}
registerProcessor('mic-processor', MicProcessor)