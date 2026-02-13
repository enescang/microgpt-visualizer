# MicroGPT Visualizer

An interactive web application for visualizing and understanding how GPT (Generative Pre-trained Transformer) models work from the ground up. Built with React and TypeScript, featuring a custom autograd engine for educational purposes.

## Features

- **Tokenizer Visualization** - See how text is broken down into tokens
- **Embeddings Explorer** - Visualize how tokens are converted to vector representations
- **Forward Pass Breakdown** - Step through the transformer's forward propagation
- **Training Process** - Watch the model learn in real-time with loss charts and heatmaps
- **Inference Mode** - Generate text and see predictions as they happen
- **Custom Autograd Engine** - Built-in automatic differentiation for transparency

## Getting Started

### Prerequisites

- Node.js 18+ 
- pnpm (recommended) or npm

### Installation

```bash
# Clone the repository
git clone https://github.com/enescang/microgpt-visualizer.git
cd microgpt-visualizer

# Install dependencies
pnpm install

# Start the development server
pnpm dev
```

The application will be available at `http://localhost:5173`

## Available Scripts

- `pnpm dev` - Start development server with hot reload
- `pnpm build` - Build for production
- `pnpm preview` - Preview production build locally
- `pnpm lint` - Run ESLint

## Project Structure

```
src/
├── components/     # Reusable UI components (Heatmap, Charts)
├── pages/          # Main application pages
│   ├── TokenizerPage.tsx
│   ├── EmbeddingsPage.tsx
│   ├── ForwardPassPage.tsx
│   ├── TrainingPage.tsx
│   └── InferencePage.tsx
├── engine/         # Core ML engine
│   ├── autograd.ts # Automatic differentiation
│   ├── model.ts    # GPT model implementation
│   ├── data.ts     # Data handling utilities
│   └── random.ts   # Random number generation
└── App.tsx         # Main application component
```

## Technology Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Custom Autograd** - Educational ML engine

## Educational Purpose

This project is designed to help understand transformer models by:

1. Providing visual feedback at each stage of the process
2. Implementing core concepts from scratch
3. Making complex operations transparent and interactive
4. Allowing experimentation with parameters in real-time

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Credits & Acknowledgments

This project is inspired by and based on:

- **[MicroGPT Guide](https://karpathy.github.io/2026/02/12/microgpt/)** by [Andrej Karpathy](https://github.com/karpathy) - Comprehensive guide explaining the implementation
- **[microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)** by Andrej Karpathy - The implementation closely mirrors this minimal GPT implementation
- **[makemore dataset](https://github.com/karpathy/makemore)** by Andrej Karpathy - Training data (names.txt) used for character-level language modeling
- **[micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy - Inspiration for the autograd engine design

Special thanks to Andrej Karpathy for creating educational resources that make deep learning accessible and understandable.

## License

MIT

## Author

[enescang](https://github.com/enescang)
