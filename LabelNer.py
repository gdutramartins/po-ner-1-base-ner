class LabelNER:
    """
        Classe preparada para conter os labels de um treinamento/predição de NER.
        Os labels podem vir sem preparação, ou seja, anotações onde somente o nome do label é informado e não estão presentes os prefixos (IOB ou BILOU).

        load_from_list e load_from_annotations devem ser utilizados no treinamento e o label set deve ser gravado utilizando save().
        No caso de predição ou carga para teste utilizar o load para carregar um label set previamente utilizado.
     
    """

    def __str__(self):
            return f"{len(self.labels_to_id)} labels {str(self.labels_to_id)}"

    def __len__(self):
        return len(self.labels_to_id)

    def __init__(self):
        self.labels_to_id = {}
        self.ids_to_label = {}
        
    def get_label_list(self):
        return self.labels_to_id.keys()

    def get_id_list(self):
        return self.ids_to_label.keys()

    def load_from_complete_list(self, labels: List[str]) -> None:
        '''
            Args: 
                labels(:obj:`List[str]`):
                Lista de labels completa, esperado conter O(utside) e os prefixos de cada entidade.
            
            Carrega a lista de labels "as is" sem tratamento.
        '''
        self.labels_to_id: Dict = {label: id for id, label in enumerate(labels)}
        self.ids_to_label:Dict = {id: label for label, id in self.labels_to_id.items()}
        self._finaliza_carga_labels()

    def load_from_simple_list(self, labels: List[str], ner_label_format: str=FORMATO_NER_IOB) -> None:
        '''
            Args: 
                labels(:obj:`List[str]`):
                    Lista de labels sem prefixo e não contendo o tipo O(utside).
                ner_label_format(:obj:`str`, `optional`, defaults to `"IOB"`):
                    Formato para classificação dos tokens de uma entidade - IOB ou BILOU
            
            Inclui o tipo O(utside) e faz a permutação entre labels e os prefixos do formato informado
        '''
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        prefix_list: str = "BI" if ner_label_format == FORMATO_NER_IOB else "BILU"

        for _num, (label, s) in enumerate(itertools.product(labels, prefix_list)):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l
        
        self._finaliza_carga_labels()

    def load_from_file(self, input_file_path:str) -> None:
        '''
            Args: 
                labelset_file(:obj:`str`):
                    Nome do arquivo contendo a lista de labels previamente gravado. Muito importante ter sido gravado por essa classe ou ter a garantia que o arquivo está com os labels na ordem correta.
            
            Carrega o arquivo com os labels ordenados. O arquivo pode ser construido manualmente, mas deve conter um label por linha, na ordem utilizada para treinar o modelo, já que essa ordem foi criada na 
            extração ou carga dos labels para o treinamento do modelo.
        '''
        with open(input_file_path, 'r' ) as label_file:
            for ind, label in enumerate(label_file):
                label = label.strip('\n')
                self.labels_to_id[label] = ind
                self.ids_to_label[ind] = label

    def _finaliza_carga_labels(self) -> None:
        ''' 
            Adicionar o label de ignorar wordpiece para os casos em que o modelo será treinado nesse formato
        '''
        self.labels_to_id[IGNORE_LABEL] = IGNORE_LABEL_MODEL_ID
        self.ids_to_label[IGNORE_LABEL_MODEL_ID] = IGNORE_LABEL

    def save(self, output_file_path: str):
        with open(output_file_path, 'w' ) as label_file:
            for label in self.labels_to_id:
                label_file.write(label)
                label_file.write('\n')    


    def convert_label_list_to_id_list(self, lista: List[str]) -> List[int]:
        """
            Converte uma lista de labels nos respectivos id`s. Para processamento no modelo essa conversão precisará ser realizada
        """
        return list(map(self.labels_to_id.get, lista))

    def convert_id_list_to_label_list(self, lista: List[int]) -> List[str]:
        """
            Converte uma lista de id`s nos respectivos labels. Para compreensão do resultado retornado pelo modelo essa conversão será necessária.
        """
        return list(map(self.ids_to_label.get, lista))  