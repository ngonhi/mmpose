dataset_info = dict(
    dataset_name='ID_card',
    paper_info=dict(
        author='Marley (Nhi Ngo)',
        title='ID Card Crop and Alignment',
        container='Techainer',
        year='2021',
        homepage='https://techainer.com/',
    ),
    keypoint_info={
        0:
        dict(name='top_left', 
            id=0, 
            color=[255, 0, 0], 
            type='upper', 
            swap='top_right'),
        1:
        dict(
            name='bottom_left',
            id=1,
            color=[0, 255, 0],
            type='lower',
            swap='bottom_right'),
        2:
        dict(
            name='bottom_right',
            id=2,
            color=[0, 255, 255],
            type='lower',
            swap='bottom_left'),
        3:
        dict(
            name='top_right',
            id=3,
            color=[255, 255, 0],
            type='upper',
            swap='top_left')
    },
    skeleton_info={
        0:
        dict(link=('top_left', 'bottom_left'), id=0, color=[128, 255, 0]),
        1:
        dict(link=('bottom_left', 'bottom_right'), id=1, color=[0, 255, 128]),
        2:
        dict(link=('bottom_right', 'top_right'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('top_right', 'top_left'), id=3, color=[255, 128, 128]),
    },
    joint_weights=[1.]*4,
    sigmas=[.025]*4,)
