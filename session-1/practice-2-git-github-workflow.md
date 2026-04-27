# Practice Git and GitHub Workflow

## Team up

Please pair up with your neighbor and follow the steps as shown in the videos.


## Videos
Chinese version:

- [GitHub Pull Request (1/3) - within one team](https://www.bilibili.com/video/BV1me4y1T73D/)
- [GitHub Pull Request (2/3) - across teams](https://www.bilibili.com/video/BV1qv4y1R7L7)
- [GitHub Pull Request (3/3) - conflict resolving](https://www.bilibili.com/video/BV1N8411V7Fa)

English version:
- [GitHub Pull Request (1/3) - within one team](https://www.bilibili.com/video/BV1LgcfzyEq7/)

- [GitHub Pull Request (2/3) - across teams](https://www.bilibili.com/video/BV1RgcfzCEpf/)

- [GitHub Pull Request (3/3) - conflict resolving](https://www.bilibili.com/video/BV1RgcfzCEpf/)

Here are three practices that you should do with your teammate, as shown in the videos. If you don't understand Chinese, that should not be a problem. You can still follow the instructions/operations shown in the videos.


### Exercise 1: Within-Team Pull Request

1. Form pairs. Each pair represents a "team" of 2 students.
2. One member (student A) create a shared repository on GitHub and add the other member (student B) as collaborators
3. Student B:
   - Clones the repository
   - Creates a feature branch
   - Makes changes (e.g., add a file with your name)
   - Pushes the branch
   - Creates a pull request
4. Student A reviews and merges Student B's PR

### Exercise 2: Cross-Team Pull Request

1. Form pairs. Each pair represents a "team" of 2 students.
2. Student A creates a repository. No need to add Student B as a collaborator.
3. Student B:
   - Fork the repository of Student A
   - Clone their fork
   - Create a feature branch
   - Make changes
   - Push to Student B's fork
   - Create a pull request to the original repository of Student A
4. Student A reviews and merge the PR

### Exercise 3: Resolving Conflicts

1. Work in pairs on a shared repository
2. Both partners create branches from the same starting point
3. Both make conflicting changes to the same file
4. One partner merges their changes first
5. The second partner then tries to merge and resolves the resulting conflict


# Video 1: Pull Requests Within One Team

This workflow is common when all team members have write access to the same repository.

### Step 1: Create a Feature Branch

Always start by creating a new branch for your feature or bugfix:

```bash
# Make sure you're on the latest version of the main branch
git checkout main
git pull origin main

# Create and switch to a new feature branch
git checkout -b feature/new-feature
```

### Step 2: Make Changes and Commit

Work on your feature, making commits as you go:

```bash
# Make your changes
# ...

# Stage and commit your changes
git add .
git commit -m "Implement new feature"
```

### Step 3: Update Your Branch with Latest Changes

Before submitting your pull request, update your branch with the latest changes from the main branch:

```bash
# Option 1: Using merge
git checkout main
git pull origin main
git checkout feature/new-feature
git merge main

# Option 2: Using rebase (creates a cleaner history)
git checkout main
git pull origin main
git checkout feature/new-feature
git rebase main
```

### Step 4: Push Your Branch to GitHub

```bash
git push origin feature/new-feature
```

### Step 5: Create a Pull Request

1. Go to the GitHub repository
2. Click on "Pull Requests" > "New Pull Request"
3. Select your feature branch as the compare branch
4. Add a title and description for your changes
5. Request reviews from team members
6. Click "Create Pull Request"

### Step 6: Review and Merge

1. Team members review the code, leaving comments or requesting changes
2. Make additional commits to address feedback if needed
3. Once approved, the pull request can be merged:
   - On GitHub: Click "Merge Pull Request"
   - Or locally:
     ```bash
     git checkout main
     git merge feature/new-feature
     git push origin main
     ```

### Step 7: Clean Up (Optional)

After your PR is merged, you can delete your feature branch:

```bash
# Delete the branch locally
git branch -d feature/new-feature

# Delete the branch on GitHub
git push origin --delete feature/new-feature
```

# Video 2. Pull Requests Across Teams (Forking Workflow)

This workflow is used when you don't have direct write access to a repository, such as when contributing to open-source projects.

### Step 1: Fork the Repository

1. Navigate to the GitHub repository you want to contribute to
2. Click the "Fork" button in the top-right corner
3. This creates a copy of the repository under your GitHub account

### Step 2: Clone Your Fork

```bash
git clone https://github.com/YOUR-USERNAME/REPOSITORY-NAME.git
cd REPOSITORY-NAME
```

### Step 3: Add the Original Repository as an Upstream Remote

```bash
git remote add upstream https://github.com/ORIGINAL-OWNER/REPOSITORY-NAME.git
git remote -v  # Verify the new remote
```

### Step 4: Create a Feature Branch

```bash
git checkout main
git pull upstream main  # Ensure your fork is up to date
git checkout -b feature/new-feature
```

### Step 5: Make Changes and Commit

```bash
# Make your changes
# ...

# Stage and commit your changes
git add .
git commit -m "Implement new feature"
```

### Step 6: Keep Your Branch Updated

```bash
# Option 1: Using merge
git checkout main
git pull upstream main
git checkout feature/new-feature
git merge main

# Option 2: Using rebase (creates a cleaner history)
git checkout main
git pull upstream main
git checkout feature/new-feature
git rebase main
```

### Step 7: Push to Your Fork

```bash
git push origin feature/new-feature
```

### Step 8: Create a Pull Request

1. Go to your fork on GitHub
2. Click on "Pull Requests" > "New Pull Request"
3. Ensure the base repository is the original repository (upstream) and the base branch is main
4. Ensure the head repository is your fork and the compare branch is your feature branch
5. Add a title and description
6. Click "Create Pull Request"

### Step 9: Address Feedback

The maintainers of the original repository will review your PR and may request changes:

```bash
# Make additional changes
git add .
git commit -m "Address feedback"
git push origin feature/new-feature
```

The PR will automatically update with your new commits.

### Step 10: After the PR is Merged

Once your PR is merged, update your fork:

```bash
git checkout main
git pull upstream main
git push origin main
```

And clean up your feature branch (optional):

```bash
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

# Video 3: Managing Merge Conflicts

Merge conflicts occur when Git cannot automatically merge changes from different branches because they modify the same part of a file.

### Resolving Conflicts

1. Open the conflicted file(s) in your editor. You'll see sections marked like this:

```
<<<<<<< HEAD
Changes from the branch you're merging into (current branch)
=======
Changes from the branch you're merging from
>>>>>>> feature-branch
```

2. Edit the file to resolve the conflict:
   - Decide which changes to keep, or how to combine them
   - Remove the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)

3. Save the file

4. Mark the conflict as resolved:

```bash
git add file.txt
```

5. Complete the merge or rebase:

```bash
# If you were merging
git commit

# If you were rebasing
git rebase --continue
```

### Using Visual Tools

Many tools can help visualize and resolve conflicts:

- **VS Code**: Built-in merge conflict resolver
- **GitLens**: VS Code extension with enhanced Git capabilities
- **Git GUI clients**: SourceTree, GitHub Desktop, etc.

## Git Merge vs. Git Rebase

Both merge and rebase integrate changes from one branch into another, but they do it differently.

### Git Merge

Merge creates a new "merge commit" that combines the changes from both branches.

```bash
git checkout main
git merge feature-branch
```

**Advantages:**
- Preserves the complete history and chronological order
- Non-destructive (doesn't change existing commits)
- Shows when and how branches were integrated

**Disadvantages:**
- Can create a cluttered history with many merge commits
- Makes the commit history non-linear

### Git Rebase

Rebase takes your commits, sets them aside, pulls in the latest changes from the target branch, and then applies your commits on top.

```bash
git checkout feature-branch
git rebase main
```

**Advantages:**
- Creates a linear, cleaner history
- Eliminates unnecessary merge commits
- Makes it easier to follow the project history

**Disadvantages:**
- Rewrites commit history (can cause problems if commits are already pushed)
- Can be more complex to handle conflicts
- Loses context of when integrations happened

### When to Use Which

- **Use Merge:**
  - For integrating completed features into the main branch
  - When you want to preserve the exact history of your branch
  - For branches that have been pushed and shared with others

- **Use Rebase:**
  - To keep your feature branch updated with the latest changes from the main branch
  - To clean up your branch before submitting a pull request
  - For local branches that haven't been shared yet
