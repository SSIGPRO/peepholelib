def map_labels(**kwargs):
    """
    Maps original labels to superclasses if a mapping_dict exists.
    """
    labels = kwargs['data']
    mapping_dict = kwargs.get('label_mapping', None)

    if mapping_dict is None:
        return labels
        
    mapped_labels = []
    for label in labels:
        for new_label, old_labels in mapping_dict.items():
            if label in old_labels:
                mapped_labels.append(new_label)
    return torch.tensor(mapped_labels)
