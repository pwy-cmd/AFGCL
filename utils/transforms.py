import albumentations as albu


def transforms(config):

    if config['aug_method'] == 'Randomcrop':
        assert config['img_q_gen'] == 'orginal_img' or config['img_q_gen'] == 'grayscale'
        train_transform = albu.Compose([
            albu.Resize(config['input_h'], config['input_w']),
            albu.RandomCrop(height=480, width=480, always_apply=True),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

        train_transform_q = albu.Compose([
            albu.Resize(config['input_h'], config['input_w']),
            albu.RandomCrop(height=256, width=256, always_apply=True),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

    if config['aug_method'] == 'all':
        train_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            albu.GaussNoise(p=0.5),
            albu.ColorJitter(p=0.5),
            albu.ToGray(p=0.5),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

        train_transform_q = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            albu.GaussNoise(p=0.5),
            albu.ColorJitter(p=0.5),
            albu.ToGray(p=0.5),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

    if config['aug_method'] == 'filp':
        train_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

        train_transform_q = albu.Compose([
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

    if config['aug_method'] == 'color':
        train_transform = albu.Compose([
            albu.ColorJitter(p=0.5),
            albu.ToGray(p=0.5),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

        train_transform_q = albu.Compose([
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

    if config['aug_method'] == 'noise':
        train_transform = albu.Compose([
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            albu.GaussNoise(p=0.5),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

        train_transform_q = albu.Compose([
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

    if config['aug_method'] == 'crop':
        assert config['img_q_gen'] == 'orginal_img' or config['img_q_gen'] == 'grayscale'
        train_transform = albu.Compose([
            albu.Resize(config['input_h'], config['input_w']),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomCrop(height=480, width=480, always_apply=True),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            albu.GaussNoise(p=0.5),
            albu.ColorJitter(p=0.5),
            albu.ToGray(p=0.5),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

        train_transform_q = albu.Compose([
            albu.Resize(config['input_h'], config['input_w']),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomCrop(height=256, width=256, always_apply=True),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            albu.GaussNoise(p=0.5),
            albu.ColorJitter(p=0.5),
            albu.ToGray(p=0.5),
            albu.Resize(config['input_h'], config['input_w']),
            albu.Normalize()])

    val_transform = albu.Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize()])

    return train_transform, train_transform_q, val_transform
