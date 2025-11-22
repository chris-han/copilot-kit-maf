import { CheckSquare as LucideCheckSquare, LucideProps } from 'lucide-react';

const Task = ({ className, ...props }: LucideProps) => {
  return <LucideCheckSquare className={className} {...props} />;
};

export default Task;