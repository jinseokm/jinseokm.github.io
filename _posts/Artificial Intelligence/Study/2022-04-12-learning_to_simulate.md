---
title: "Learning to Simulate Complex Physics with Graph Networks"
excerpt: "Learning to Simulate"
categories:
  - Study
tags: [GNN, Deep Learning, Physics]
toc: true
toc_sticky: true
toc_label: "On this page"
published: true
use_math: true

date: 2022-04-12
last_modified_at: 2022-04-14
---

# [Learning to Simulate Complex Physics with Graph Networks (ICML 2020)](https://arxiv.org/abs/2002.09405)

## Graph Neural Network (GNN)
먼저, 그래프가 무엇인지 알아보자. 그래프는 점들과 그 점들을 잇는 선으로 이루어진 데이터 구조이다. 관계나 상호작용을 나타내는 데이터를 분석할 때 주로 쓰인다. 대표적인 예로는 페이스북 친구관계, 왓챠플레이(유튜브, 넷플릭스) 유저-영상 감상여부 등이 있다.

<center>
<figure style="width: 30%"> <img src="/images/Study/learning_to_simulate/graph.png" alt="Exmaple of Graph"/>
<figcaption>점과 선으로 이루어진 그래프</figcaption>
</figure>

<figure style="width: 80%"> <img src="/images/Study/learning_to_simulate/relations.jpg" alt="Relation Networks"/>
<figcaption>인간 관게도</figcaption>
</figure>
</center>


GNN은 2009년에 제안된 기법(Scarselli et al., 2009)으로, 노드와 노드 사이의 관계, 엣지의 가중치 들을 고려하여 그래프로 네트워크를 구성하는 방법이다. 

<center>
<figure style="width: 90%"> <img src="/images/Study/learning_to_simulate/gnn.png" alt="GNN"/>
<figcaption>Graph Neural Network</figcaption>
</figure>
</center>

## Graph Networks
Graph Networks 는 DeepMind에서 제작한 python, tf 기반 GNN 라이브러리다.
그래프를 입력받고 마찬가지로 그래프를 출력하는 구성이며 입력 그래프는 엣지(E), 노드(V), 그리고 전역 파라미터(u)로 구성되어있고, 출력 그래프는 입력 그래프와 같은 형태이나 각 파라미터들을 업데이트한 상태가 된다. 또한, 노드, 엣지, 글로벌에 더해 Sender/Reciever 자료형을 가지고 있다. 

<center>
<figure style="width: 80%"> <img src="/images/Study/learning_to_simulate/graphnets.jpg" alt="GNN"/>
<figcaption>Graph Networks</figcaption>
</figure>
</center>

아래 그림과 같은 그래프가 있을때, `[Sender, Receiver]` 는 `[[0,0], [0,1], [1,0], [1,0]]` 와 같은 식으로 구성되어 있다. 이 때, 각 원소의 인덱스는 엣지의 인덱스를 나타낸다.

<center>
<figure style="width: 50%"> <img src="/images/Study/learning_to_simulate/graph-example.jpg" alt="Graph Example"/>
<figcaption>Graph Example</figcaption>
</figure>
</center>

입출력 네트워크를 구성하는 여러가지 방법이 있다. 

<center>
<figure style="width: 80%"> <img src="/images/Study/learning_to_simulate/modules.jpg" alt="Graph Example"/>
<figcaption>Graph Networks Modules</figcaption>
</figure>
</center>

## Learning to Simulate 코드 뜯어보기
DeepMind에서는 Graph와 Particle-based Simulation을 결합한 `Graph Network-based Simulators (GNS)` 를 제안했다. Learning to Simulate를 이용해서 시뮬레이션을 모사하면 다음과 같이 유사하게 예측을 할 수 있다.

<center>
<figure> <img src="/images/Study/learning_to_simulate/water_ramps_rollout.gif" alt="Learning to Simulate Example"/>
<figcaption>Learning to Simulate 모델 예시</figcaption>
</figure>
</center>

[Github](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)에 소스코드를 오픈하여 공유하고 있다. 파일 구조는 다음과 같다.

> - train.py: 전체적으로 학습/평가/시뮬레이션 모사를 담당
> - learned_simluator.py: 한스텝을 학습, 다음 포지션을 예측. 데이터 전처리, 정규화  기능 포함
> - graph_network.py: 코어 네트워크 모델
> - render_rollout.py: 결과 가시화
> - {noise/connectivity/reading}_utils.py: 노이즈 첨가, 그래프 연결정보 구성 (주변입자 탐색), 데이터 셋 읽기 등의 유틸 기능
 
초기 세팅은 flags를 이용해서 다음과 같이 정의하여 argument들을 사용한다.

```python
# train.py

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')


FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
```

`LearnedSimulator`는 시뮬레이터와 GNN을 엮어주는 클래스이다. 이 클래스를 통해서 전체적인 학습/평가/시뮬레이션을 수행한다. 소스코드 원문에는 친절하게 documentation이 작성되어있지만 아래 코드들에서는 가시성을 위해 지워두었다. 

```python
# learned_simluator.py

class LearnedSimulator(snt.AbstractModule):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
      self,
      num_dimensions,
      connectivity_radius,
      graph_network_kwargs,
      boundaries,
      normalization_stats,
      num_particle_types,
      particle_type_embedding_size,
      name="LearnedSimulator"):
    super().__init__(name=name)

    self._connectivity_radius = connectivity_radius
    self._num_particle_types = num_particle_types
    self._boundaries = boundaries
    self._normalization_stats = normalization_stats
    with self._enter_variable_scope():
      self._graph_network = graph_network.EncodeProcessDecode(
          output_size=num_dimensions, **graph_network_kwargs)

      if self._num_particle_types > 1:
        self._particle_type_embedding = tf.get_variable(
            "particle_embedding",
            [self._num_particle_types, particle_type_embedding_size],
            trainable=True, use_resource=True)
```
위 코드의 LearnedSimulator 클래스는 멤버 변수로 `_graph_network` 를 갖는데, 이 변수는 Encoder - Processor - Decoder 로 구성되는 코어 네트워크를 정의한 `EncodeProcessDecode` 클래스이다. 클래스의 세부 네트워크는 다음과 같다.

```python
# graph_network.py

class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode function approximator for learnable simulator."""

  def __init__(
      self,
      latent_size: int,
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      output_size: int,
      reducer: Reducer = tf.math.unsorted_segment_sum,
      name: str = "EncodeProcessDecode"):

    super().__init__(name=name)

    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._output_size = output_size
    self._reducer = reducer

    with self._enter_variable_scope():
      self._networks_builder()
```

`_networks_builder()` 함수는 다음과 같다. 노드와 엣지를 MLP로 엮어서 독립적인 Encoder 네트워크를 각각 만들고, `_num_message_passing_steps` 만큼의 process 네트워크를 구축한다. hidden layer를 쌓는다고도 볼 수 있다. 이후 process 과정을 통해 얻어진 latent vector들을 decoder 과정을 통해 초기 노드와 동일한 차원으로 만들어준다.

```python
# graph_network.py
# class EncodeProcessDecode(snt.AbstractModule):

    def _networks_builder(self):
    """Builds the networks."""

        def build_mlp_with_layer_norm():
        mlp = build_mlp(
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._latent_size)
        return snt.Sequential([mlp, snt.LayerNorm()])

        # The encoder graph network independently encodes edge and node features.
        encoder_kwargs = dict(
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm)
        self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)

        # Create `num_message_passing_steps` graph networks with unshared parameters
        # that update the node and edge latent features.
        # Note that we can use `modules.InteractionNetwork` because
        # it also outputs the messages as updated edge latent features.
        self._processor_networks = []
        for _ in range(self._num_message_passing_steps):
        self._processor_networks.append(
            gn.modules.InteractionNetwork(
                edge_model_fn=build_mlp_with_layer_norm,
                node_model_fn=build_mlp_with_layer_norm,
                reducer=self._reducer))

        # The decoder MLP decodes node latent features into the output size.
        self._decoder_network = build_mlp(
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size)
```

주변 입자 탐색에는 [scikit-learn의 K-D Tree 알고리즘](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html)을 사용한다. 

```python
# connectivity_utils.py

def _compute_connectivity(positions, radius, add_self_edges):
  """Get the indices of connected edges with radius connectivity.

  Args:
    positions: Positions of nodes in the graph. Shape:
      [num_nodes_in_graph, num_dims].
    radius: Radius of connectivity.
    add_self_edges: Whether to include self edges or not.

  Returns:
    senders indices [num_edges_in_graph]
    receiver indices [num_edges_in_graph]

  """
  tree = neighbors.KDTree(positions)
  receivers_list = tree.query_radius(positions, r=radius)
  num_nodes = len(positions)
  senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
  receivers = np.concatenate(receivers_list, axis=0)

  if not add_self_edges:
    # Remove self edges.
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]

  return senders, receivers
```

논문에서는 이 부분이 CPU에서 수행되고, 전체 플로우에서 큰 부분을 차지하여, 성능이 떨어진다고 언급했다. 실제로 Water-3D 케이스의 시뮬레이션 1 iteration에 Simulator는 0.104s, GNS의 경우 0.358s로, Simulator 대비 345% 의 속도를 얻었다고 한다. 여기서 주변입자 탐색을 제외한 Simulation에 수행된 시간은 0.071s로 19.8%였다. 이는 주변입자탐색 알고리즘을 고도화/최적화 하면 더욱 빨라질 가능성이 있음을 의미한다.