import torch
import torch.nn as nn

def train_honest_client(model_cls, input_dim, global_weights, client_data, batch_size=32, local_epochs=5, lr=0.01, seed=None):
    """
    Train an honest client for local_epochs using fast tensor indexing SGD.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model = model_cls(input_dim)
    model.load_state_dict(global_weights)
    model.train()
    
    x_train, y_train = client_data['x'], client_data['y']
    num_samples = len(x_train)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    
    for epoch in range(local_epochs):
        if generator is not None:
            perm = torch.randperm(num_samples, generator=generator)
        else:
            perm = torch.randperm(num_samples)
            
        for start_idx in range(0, num_samples, batch_size):
            batch_idx = perm[start_idx:start_idx + batch_size]
            bx, by = x_train[batch_idx], y_train[batch_idx]
            
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    global_vec = torch.cat([p.data.view(-1) for p in global_weights.values()])
    local_vec = torch.cat([p.data.view(-1) for p in model.parameters()])
    local_update = local_vec - global_vec
    
    return local_update, num_samples
