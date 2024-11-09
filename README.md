# Build frontend
cd frontend
npm install
npm run build

# Deploy to Azure
cd ../backend
zip -r ../app.zip .
az webapp deploy --resource-group your_rg --name your_app --src-path ../app.zip