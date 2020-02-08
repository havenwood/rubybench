# from bryanbibat's gist: https://gist.github.com/2348802

class Synapse
  WEIGHT_RANGE = (-1.0..1.0).freeze

  attr_accessor :weight, :prev_weight
  attr_reader :source_neuron, :dest_neuron

  def initialize(source_neuron:, dest_neuron:)
    @source_neuron = source_neuron
    @dest_neuron = dest_neuron
    @prev_weight = @weight = rand(WEIGHT_RANGE)
  end
end

class Neuron
  LEARNING_RATE = 1.0
  MOMENTUM = 0.3
  WEIGHT_RANGE = (-1.0..1.0).freeze

  attr_reader :synapses_in, :synapses_out
  attr_reader :threshold, :prev_threshold, :error
  attr_accessor :output

  def initialize
    @synapses_in = []
    @synapses_out = []
    @prev_threshold = @threshold = rand(WEIGHT_RANGE)
  end

  def calculate_output
    activation = synapses_in.sum do |synapse|
      synapse.weight * synapse.source_neuron.output
    end - @threshold

    @output = 1.fdiv(Math.exp(-activation) + 1)
  end

  def output_train(rate:, target:)
    @error = (target - @output) * derivative
    update_weights(rate)
  end

  def hidden_train(rate:)
    @error = synapses_out.sum do |synapse|
      synapse.prev_weight * synapse.dest_neuron.error
    end * derivative
    update_weights(rate)
  end

  private

  def derivative
    @output * (1 - @output)
  end

  def update_weights(rate)
    update_synapses_in(rate)
    update_thresholds(rate)
  end

  def update_synapses_in(rate)
    @synapses_in.each do |synapse|
      prev_weight = synapse.weight
      synapse.weight += LEARNING_RATE * rate * @error * synapse.source_neuron.output
      synapse.weight += MOMENTUM * (synapse.weight - synapse.prev_weight)
      synapse.prev_weight = prev_weight
    end
  end

  def update_thresholds(rate)
    prev_threshold = @threshold
    @threshold += LEARNING_RATE * rate * @error * -1
    @threshold += MOMENTUM * (@threshold - @prev_threshold)
    @prev_threshold = prev_threshold
  end
end

class NeuralNetwork
  def initialize(inputs:, hidden:, outputs:)
    @input_layer  = Array.new(inputs) { Neuron.new }
    @hidden_layer = Array.new(hidden) { Neuron.new }
    @output_layer = Array.new(outputs) { Neuron.new }

    add_synapses(@input_layer, @hidden_layer)
    add_synapses(@hidden_layer, @output_layer)
  end

  def train(inputs:, targets:)
    feed_forward(inputs)

    @output_layer.zip(targets).each do |neuron, target|
      neuron.output_train(rate: 0.3, target: target)
    end
    @hidden_layer.each { |neuron| neuron.hidden_train(rate: 0.3) }
  end

  def feed_forward(inputs)
    @input_layer.zip(inputs) { |neuron, input| neuron.output = input }
    @hidden_layer.each(&:calculate_output)
    @output_layer.each(&:calculate_output)
  end

  def current_outputs
    @output_layer.map(&:output)
  end

  private

  def add_synapses(source_layer, dest_layer)
    source_layer.product(dest_layer) do |source_neuron, dest_neuron|
      synapse = Synapse.new(source_neuron: source_neuron,
                            dest_neuron: dest_neuron)

      source_neuron.synapses_out << synapse
      dest_neuron.synapses_in << synapse
    end
  end
end

require 'benchmark'

ARGV.fetch(0) { 5 }.to_i.times do
  results = Benchmark.measure do
    xor = NeuralNetwork.new(inputs: 2, hidden: 10, outputs: 1)

    10_000.times do
      xor.train(inputs: [0, 0], targets: [0])
      xor.train(inputs: [1, 0], targets: [1])
      xor.train(inputs: [0, 1], targets: [1])
      xor.train(inputs: [1, 1], targets: [0])
    end

    xor.feed_forward([0, 0])
    puts xor.current_outputs
    xor.feed_forward([0, 1])
    puts xor.current_outputs
    xor.feed_forward([1, 0])
    puts xor.current_outputs
    xor.feed_forward([1, 1])
    puts xor.current_outputs
  end

  puts results
end

