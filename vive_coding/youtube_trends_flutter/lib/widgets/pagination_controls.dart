import 'package:flutter/material.dart';

class PaginationControls extends StatelessWidget {
  final int currentPage;
  final int totalPages;
  final void Function(int) onPageChanged;
  
  const PaginationControls({
    super.key, 
    required this.currentPage, 
    required this.totalPages, 
    required this.onPageChanged,
  });
  
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: currentPage > 1
                ? () => onPageChanged(-1)
                : null,
          ),
          Text('$currentPage / $totalPages'),
          IconButton(
            icon: const Icon(Icons.arrow_forward),
            onPressed: currentPage < totalPages
                ? () => onPageChanged(1)
                : null,
          ),
        ],
      ),
    );
  }
} 