import json
from git import Repo, InvalidGitRepositoryError

class DirtyGitRepositoryError(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)

def get_git_hash(directory, length=10):
    '''
    This function will check the state of the git repository.

    * If there is no repo, an InvalidGitRepositoryError is raised.
    * If the repo is dirty, a DirtyRepositoryException is raised.
    * If none of the above applies, the hash of the HEAD commit is returned.

    Parameters
    ----------
    directory: str
        The path to the directory to check.
    length: int
        The number of characters of the hash to return (default 10).
    '''

    # Check the state of the github repository
    repo = Repo(directory, search_parent_directories=True)
    if repo.is_dirty():
        raise DirtyGitRepositoryError('The git repo has uncommited modifications. Aborting simulation.')
    else:
        return repo.head.commit.hexsha[:length]


def json_append(filename, entry):
    '''
    This function incrementally add entries to a json file
    while keeping the format correct

    Parameters
    ----------
    filename: str
        the name of the JSON file
    entry:
        the new entry to append
    '''
    import json

    with open(filename, 'at') as f:

        if f.tell() == 0:
            # first write, add array
            json.dump([entry], f, indent=0)

        else:
            # remove last character ']' and '\n'
            f.seek(f.tell() - 2, 0)
            f.truncate()

            # add missing comma to previous element
            f.write(',\n')

            # dump the latest entry
            json.dump(entry, f, indent=0)

            # close the json file
            f.write('\n]')


