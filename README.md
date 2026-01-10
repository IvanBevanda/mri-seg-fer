# mri-seg-fer
Segmentation of MRI images to identify brain tumors.

# Installation and usage guide

## Clone repository
Note that this guide is intended for Windows users on the team. If you're using Linux, the same general steps shouldn't change, but the process will be quite different.

Feel free to reach out to me if you encounter issues or feel overwhelmed.

Perform the following steps:

1. Install [Git](https://git-scm.com/install/windows).
2. Create a [GitHub account](https://github.com/) with the official @fer.hr address.
3. Clone this repository. [Guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?platform=windows&tool=webui). This part may be tricky due to authentication issues.
4. Create empty `data/` and `output/` folders inside the repository.
5. Change into this repository: `cd mri-seg-fer`.
6. Install everything from `requirements.txt`: `py -m pip install -r requirements.txt`.

## Making changes
0. Enter the repository folder `cd mri-seg-fer`.
1. Switch to main branch. Run `git checkout main`.
2. Pull remote changes: `git pull`.
3. Create a new branch: `git checkout -b my_super_cool_new_feature`.
4. Code away.
5. When done, format (make it look pretty) it using black: `black .`.
6. Commit your changes: `git add .` and then `git commit . -m "Short informative message about what you did"`.
7. Check that all is done: run `git status` -- it should say `nothing to commit`.
8. Push your local branch to GitHub: `git push -u origin my_super_cool_new_feature`.
9. Create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request#creating-the-pull-request) -- specify `main` as the target branch you'd like to merge into.
10. Wait for review/approval.

If you encounter issues due to authentication, feel free to contact me, and we'll set up SSH keys and such so you don't have to worry.

# Useful tips:

There's a `scripts/` folder containing `populate_data.py` which will create `train_X.npy`, `train_Y.npy`, `test_X.npy` and `test_Y.npy` filled with useless values, but of a proper shape. Useful for testing purposes. To use it just run `py -m scripts/populate_data.py`.

# Defined interface and tasks:

- [x] Create a GitHub repo with the project skeleton in place.
- [ ] Implement the UNet model -- inside `models.py`. 
    - It should output in the form `(BATCH_SIZE, 2, WIDTH, HEIGHT)` and take inputs of the form `(BATCH_SIZE, 1, WIDTH, HEIGHT)`.
- [ ] Implement the validation loop -- inside `training.py`, specifically `eval_model()`.
- [ ] Implement preprocessing -- inside `source/preprocessing/`. 
    - It should create `train_X.npy`, `train_Y.npy`, `test_X.npy`, `test_Y.npy`. 
    - `X`s have shape `(1, WIDTH, HEIGHT)`, and `Y`s have shape `(1, WIDTH, HEIGHT)` -- the same, but different from model output (because of `CrossEntropyLoss`).
    - Find an appropriate dataset with *one channel* inputs and *one channel* outputs -- check with me if unsure.
- [x] Create a training loop.
- [ ] Implement postprocessing -- inside `source/postprocessing`.
    - Find good segmentation metrics and calculate them for the trained model on the test dataset.
    - Show a few examples -- run them through the model and display them.
    - Plot training and testing losses over the course of traininig. They will be arrays, indexed by epoch, inside `output/training_losses.npy` and `output/testing_losses.npy`.

# Notes

If reusing code, add the source to `references.txt`.